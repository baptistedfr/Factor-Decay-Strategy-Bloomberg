import pandas as pd
import numpy as np
from portfolio import BasePortfolio, FractilePortfolio, PureFactorPortfolio
from datetime import datetime
from dateutil.relativedelta import relativedelta
from Utilities import get_rebalancing_dates
import logging
import plotly.express as px
from enum import Enum
from tqdm import tqdm
from enums import Universe, FrequencyType
from results import Results
logging.basicConfig(level=logging.INFO)



class PortfolioAnalysis:

    def __init__(self, universe : Universe):
        """
        Parameters:
        ------------
        universe : Universe
        """

        # Vérifier que les fichiers existent avant de les charger
        required_files = ['compo', 'Price', 'ROE', 'Price To Book']
        for key in required_files:
            if universe.value.get(key) is None:
                raise FileNotFoundError(f"Le fichier pour '{key}' est manquant dans le paramètre 'universe'.")
            
        self.universe_data = {}
        for key in required_files:
            try:
                self.universe_data[key] = pd.read_parquet(f"Input_parquet/{universe.value[key]}")
            except Exception as e:
                raise IOError(f"Erreur lors du chargement de {key} : {e}")

    
    def compare_strategies(self, target_factor: str, 
                            sensi_factors: list[str], 
                            start_date_str : str,
                            end_date_str : str, 
                            Portfolio : BasePortfolio = FractilePortfolio,
                            rebalance_type : FrequencyType = FrequencyType.MONTHLY,
                            transaction_fees : float = 0.0005):
        
        # Convertir les dates en objets datetime
        start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # Étape 1 : Récupérer les tickers communs et filtrer les données
        common_tickers = self.get_tickers_in_range(start_date_dt, end_date_dt)
        universe_filtered = self.filter_universe_data(start_date_dt, end_date_dt, common_tickers)

        factor_df = self.__construct_dataframe_factors(universe_filtered, sensi_factors, start_date_dt)

        all_dates = sorted(set(factor_df["Date"].tolist()))
        portfolio = Portfolio(factor_df, target_factor, sensi_factors)

        # 🔸 Stratégie 1 : Rebalancement périodique

        results_basic_strat = self.compute_strategy_performance(portfolio, 
                                                                factor_df, 
                                                                universe_filtered, 
                                                                target_factor,
                                                                all_dates, 
                                                                rebalance_type,
                                                                f"{target_factor} Basic strategy {Portfolio.__name__}",
                                                                fees = transaction_fees)
        
        # Stratégie 2 : Rebalancement à mi-période (half-date)
        results_half_date_strat = self.compute_strategy_performance(portfolio,
                                                                    factor_df,
                                                                    universe_filtered,
                                                                    target_factor,
                                                                    all_dates,
                                                                    FrequencyType.HALF_EXPOSURE,
                                                                    f"{target_factor} Half life strategy {Portfolio.__name__}",
                                                                    fees = transaction_fees)
    
        combined_results = [results_basic_strat, results_half_date_strat]
        
        if Portfolio.__name__ == "PureFactorPortfolio":
            # Stratégie 3 : Rebalancement sur exposition non désirée
            results_undesired_strat = self.compute_strategy_performance(portfolio,
                                                                        factor_df,
                                                                        universe_filtered,
                                                                        target_factor,
                                                                        all_dates,
                                                                        FrequencyType.UNDESIRED_EXPOSURE,
                                                                        f"{target_factor} Undesired strategy {Portfolio.__name__}",
                                                                        fees = transaction_fees)
            combined_results.append(results_undesired_strat)
        return Results.compare_results(combined_results), combined_results
    
    
    def compute_strategy_performance(self, portfolio: BasePortfolio, 
                                  factor_df: pd.DataFrame, 
                                  universe_df: dict, 
                                  factor: str, 
                                  all_dates: list, 
                                  rebalance_type: FrequencyType = FrequencyType.MONTHLY,
                                  strategy_name : str= "",
                                  fees: float = 0.0005):
        """
        Calcule la performance d'une stratégie en fonction du type de rebalancement.
        """
        strat_value = 100
        total_fees = 0
        stored_values = [strat_value]
        weights_dict = {}

        returns_df = universe_df['Returns']
        actual_tickers = self.get_tickers_in_range(all_dates[0])

        # Premier rebalancement
        returns_ptf = returns_df.loc[returns_df['Date'] <= all_dates[0], ['Date'] + list(actual_tickers)]
        df_subset = factor_df.loc[(factor_df["Date"] == all_dates[0]) & (factor_df['Ticker'].isin(list(actual_tickers))), :]
        initial_ptf, ptf_sensi = portfolio.construct_portfolio(
            df_subset,
            rebalance_weight=True,
            returns=returns_ptf
        )
        weights = dict(zip(initial_ptf['Ticker'], initial_ptf['Weight']))
        weights_dict[all_dates[0]] = weights

        # Calcul de l'exposition initiale
        initial_exposure = ptf_sensi[factor]
        undesired_exposure = ptf_sensi.drop(factor).abs().sum()
        
        if rebalance_type == FrequencyType.MONTHLY:
            # Si c'est un rebalancement mensuel, on vérifie la date de rebalancement
            rebalancing_dates = get_rebalancing_dates(all_dates, FrequencyType.MONTHLY)
        
        rebalancing_dt = []

        for t in tqdm(range(1, len(all_dates)), desc=f"Running Backtesting {strategy_name}", leave = False):
        #for t in range(1, len(all_dates)):
            returns_dict = returns_df.loc[returns_df['Date'] == all_dates[t], list(actual_tickers)].squeeze().to_dict()
            prev_weights = np.array([weights_dict[all_dates[t-1]][ticker] for ticker in actual_tickers])

            daily_returns = np.array([returns_dict.get(ticker, 0) for ticker in actual_tickers])
            return_strat = np.dot(prev_weights, daily_returns)
            new_strat_value = strat_value * (1 + return_strat)

            # Rebalancement selon le type spécifié
            if rebalance_type == FrequencyType.MONTHLY and t in rebalancing_dates:

                rebalancing_dt.append(all_dates[t])
                actual_tickers = self.get_tickers_in_range(all_dates[t])
                df_ptf, ptf_sensi, new_weights = self.rebalance_strat(actual_tickers, all_dates[t], factor_df, returns_df, portfolio)

                transaction_costs = self.calculate_transaction_costs(weights, new_weights, fees)
                total_fees+=transaction_costs
                new_strat_value -= strat_value * transaction_costs
                

            elif rebalance_type == FrequencyType.HALF_EXPOSURE:
                # Si c'est un rebalancement basé sur l'exposition
                current_exposure = ptf_sensi[factor]

                if current_exposure <= initial_exposure / 2:
                    rebalancing_dt.append(all_dates[t])
                    actual_tickers = self.get_tickers_in_range(all_dates[t])
                    df_ptf, ptf_sensi, new_weights = self.rebalance_strat(actual_tickers, all_dates[t], factor_df, returns_df, portfolio)
                    initial_exposure = ptf_sensi[factor]

                    transaction_costs = self.calculate_transaction_costs(weights, new_weights, fees)
                    total_fees+=transaction_costs
                    new_strat_value -= strat_value * transaction_costs
                else:
                    # Si l'exposition n'a pas été divisée par deux, on applique simplement un drift aux poids et on recalcule les expos
                    df_subset = factor_df.loc[(factor_df["Date"] == all_dates[t]) & (factor_df['Ticker'].isin(list(actual_tickers))), :]
                    df_subset['Weight'] = df_subset['Ticker'].map(weights)
                    df_ptf, ptf_sensi = portfolio.construct_portfolio(df_subset, rebalance_weight=False, returns=False)
                    new_weights = {ticker: weights[ticker] * (1 + returns_dict[ticker]) for ticker in weights}
            elif rebalance_type == FrequencyType.UNDESIRED_EXPOSURE:
                # Nouvelle stratégie : Rebalancer si la somme des expositions aux autres facteurs dépasse celle du target_factor
                undesired_exposure = ptf_sensi.drop(factor).abs().sum()
                target_factor_exposure = ptf_sensi[factor]

                if undesired_exposure >= target_factor_exposure:
                    rebalancing_dt.append(all_dates[t])
                    actual_tickers = self.get_tickers_in_range(all_dates[t])
                    df_ptf, ptf_sensi, new_weights = self.rebalance_strat(actual_tickers, all_dates[t], factor_df, returns_df, portfolio)

                    transaction_costs = self.calculate_transaction_costs(weights, new_weights, fees)
                    total_fees += transaction_costs
                    new_strat_value -= strat_value * transaction_costs
                else:
                    # Si l'exposition non désirée n'est pas supérieure, on applique simplement un drift aux poids et on recalcule les expos
                    df_subset = factor_df.loc[(factor_df["Date"] == all_dates[t]) & (factor_df['Ticker'].isin(list(actual_tickers))), :]
                    df_subset['Weight'] = df_subset['Ticker'].map(weights)
                    df_ptf, ptf_sensi = portfolio.construct_portfolio(df_subset, rebalance_weight=False, returns=False)
                    new_weights = {ticker: weights[ticker] * (1 + returns_dict[ticker]) for ticker in weights}            
            else:
                new_weights = {ticker: weights[ticker] * (1 + returns_dict[ticker]) for ticker in weights}
            
            # Stockage des nouveaux poids et valeurs
            weights_dict[all_dates[t]] = new_weights
            stored_values.append(new_strat_value)

            weights = new_weights
            strat_value = new_strat_value

        return self.output(f"{strategy_name}", stored_values, weights_dict, all_dates, rebalancing_dt, total_fees)

    def rebalance_strat(self, actual_tickers : list, date : datetime, 
                        factor_df : pd.DataFrame, returns_df : pd.DataFrame, 
                        portfolio : BasePortfolio) -> tuple:
        """
        Effectue le rebalancement du portefeuille à une date donnée.

        Args:
            actual_tickers (list): Liste des tickers actifs à la date donnée.
            date (datetime): Date du rebalancement.
            factor_df (pd.DataFrame): DataFrame contenant les facteurs.
            returns_df (pd.DataFrame): DataFrame contenant les rendements.
            portfolio (BasePortfolio): Instance du portefeuille.

        Returns:
            tuple: (df_ptf, ptf_sensi, new_weights)
        """
        # Filtrer factor_df pour la date et les tickers actifs
        df_subset = factor_df.loc[(factor_df["Date"] == date) & (factor_df['Ticker'].isin(list(actual_tickers))), :]
        
        # Vérifier que la date existe dans returns_df et récupérer son index
        index_date = int(returns_df.loc[returns_df['Date'] == date, :].index[0])
        
        # Récupérer les rendements sur une période de 252 jours avant la date
        returns_ptf = returns_df.loc[
            (returns_df.index >= index_date - 252) & (returns_df.index <= index_date), 
            ['Date'] + list(actual_tickers)
        ]
        
        # Construire le portefeuille avec les nouveaux poids
        df_ptf, ptf_sensi = portfolio.construct_portfolio(
            df_subset,
            rebalance_weight=True,
            returns=returns_ptf
        )
        
        # Extraire les nouveaux poids sous forme de dictionnaire
        new_weights = dict(zip(df_ptf['Ticker'], df_ptf['Weight']))
        
        return df_ptf, ptf_sensi, new_weights

    
    
    def output(self, strategy_name : str, stored_values : list[float], stored_weights : list[float], 
               dates : list, rebalancing_dates : list , 
               fees : float = 0, frequency_data : FrequencyType = FrequencyType.DAILY) -> Results :
        """Create the output for the strategy and its benchmark if selected
        
        Args:
            stored_values (list[float]): Value of the strategy over time
            stored_weights (list[float]): Weights of every asset in the strategy over time
            strategy_name (str) : Name of the current strategy

        Returns:
            Results: A Results object containing statistics and comparison plot for the strategy (& the benchmark if selected)
        """

        ptf_weights = pd.DataFrame(stored_weights).T
        ptf_values = pd.Series(stored_values, index=dates)
        ptf_rebalacing = pd.Series([1 if date in rebalancing_dates else 0 for date in dates], index=dates)


        results_strat = Results(ptf_values=ptf_values, ptf_weights=ptf_weights, 
                                ptf_rebalancing=ptf_rebalacing, total_fees = fees,
                                strategy_name=strategy_name, data_frequency=frequency_data)
        
        results_strat.get_statistics()
        results_strat.create_plots()

        return results_strat
    
    def get_factor_information(self, target_factor: str, 
                               sensi_factors: list[str], 
                               computation_date_str : str, 
                               Portfolio : BasePortfolio = FractilePortfolio, 
                               plot = True):
        """
        Construct the portfolio at the first date and get the sensi at each following date.
        Args:
            target_factor (str): Type of long-short portfolio to create (Momentum, Value, Growth ...)
            sensi_factors (list[str], optional): List of factors for which a sensitivity is calculated
            computation_date (str): Date to compute the analysis
        """
        computation_date_dt = datetime.strptime(computation_date_str, "%Y-%m-%d")
        common_tickers = self.get_tickers_in_range(computation_date_dt)

        universe_filtered = self.filter_universe_data(computation_date_dt, None, common_tickers)
    
        returns_df = universe_filtered['Returns']

        factor_df = self.__construct_dataframe_factors(universe_filtered, sensi_factors, computation_date_dt)

        all_dates = sorted(set(factor_df["Date"].tolist()))

        ptf_construction : BasePortfolio = Portfolio(df_factor=factor_df, target_factor=target_factor, sensi_factors=sensi_factors)
        initial_ptf, first_sensi = ptf_construction.construct_portfolio(factor_df.loc[factor_df["Date"] == all_dates[0], :], 
                                                                        rebalance_weight=True, 
                                                                        returns = returns_df.loc[returns_df['Date'] <= all_dates[0], :])
        # Keep in memory the weights optimized for the given factor
        weights = initial_ptf["Weight"].tolist()

        # Compute portfolio sensibility to sensi factors for all dates
        sensi_records = [{"Date": factor_df['Date'].iloc[0], **first_sensi.to_dict()}]
        for date in all_dates[1:]:
            # Add the initial weights to the dataframe at the current date
            df_date = factor_df[factor_df["Date"] == date].copy()
            df_date["Weight"] = weights
            # Compute the sensi
            _, sensi_date = ptf_construction.construct_portfolio(df_date, 
                                                                 rebalance_weight=False,
                                                                 returns = None)
            sensi_records.append({"Date": date, **sensi_date.to_dict()})


        sensi_historic = pd.DataFrame(sensi_records)
        half_life_date, half_life_days = self.compute_half_life_factor(sensi_historic, target_factor) 
        
        if plot:
            print(f"Période d'analyse : {sensi_historic['Date'].iloc[0]} -> {sensi_historic['Date'].iloc[-1]}")
            self.plot_sensibilities(sensi_historic, target_factor, computation_date_str, half_life_date)

        return sensi_historic, half_life_days
    
    def filter_universe_data(self, start_date_dt, end_date_dt, common_tickers):
        """
        Filtre les données de l'univers en fonction des dates et des tickers communs.
        """
        start_date, end_date = self.get_analysis_dates(self.universe_data['Price'], start_date_dt, end_date=end_date_dt)
        universe_filtered = {
            key: df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), ['Date'] + list(common_tickers)]
            for key, df in self.universe_data.items() if key != 'compo'
        }
        # Add the Market
        market_df = self.universe_data['Price'].loc[
            (self.universe_data['Price']['Date'] >= start_date) & 
            (self.universe_data['Price']['Date'] <= end_date), 
            ['Date', 'Index']
        ]
        universe_filtered['Market'] = market_df
        universe_filtered['Returns'] = universe_filtered['Price'].set_index("Date").pct_change().dropna().reset_index('Date')
        return universe_filtered

    @staticmethod
    def __construct_dataframe_factors(universe_filtered, sensi_factors: list[str], computation_date_dt : datetime):
        """
        Calcule les facteurs demandés et retourne un DataFrame avec une ligne par ticker/date.
        
        Args:
            sensi_factors (list[str]): Liste des facteurs à calculer.
        
        Returns:
            pd.DataFrame: DataFrame structuré avec une colonne par facteur.
        """
        # Récupérer les prix et aligner les autres datasets
        price_df = universe_filtered['Price'].set_index("Date")
        returns_df = universe_filtered['Returns'].set_index("Date")

        # Initialiser la structure du DataFrame final
        factor_df = pd.DataFrame()
        factor_df["Date"] = np.repeat(price_df.index, len(price_df.columns))
        factor_df["Ticker"] = np.tile(price_df.columns, len(price_df))

        # Ajouter chaque facteur comme une colonne
        if "Value" in sensi_factors:
            # print("Calcul du facteur Value (P/B)...")
            ptb_df = universe_filtered['Price To Book'].set_index("Date").reindex(price_df.index).ffill()
            factor_df["Value"] = - ptb_df.values.flatten()

        if "Momentum" in sensi_factors:
            # print("Calcul du facteur Momentum (12-1 month return)...")
            P_t_12 = price_df.shift(252)
            P_t_1 = price_df.shift(21)
            factor_df["Momentum"] = (P_t_1 / P_t_12 - 1).values.flatten()

        if "Quality" in sensi_factors:
            # print("Calcul du facteur Quality (ROE)...")
            roe_df = universe_filtered['ROE'].set_index("Date").reindex(price_df.index).ffill()
            factor_df["Quality"] = roe_df.values.flatten()

        if "Low Volatility" in sensi_factors:
            # print("Calcul du facteur Low Volatility (252-day vol)...")
            rolling_vol = (price_df.pct_change().rolling(window=252).std())*np.sqrt(252)
            factor_df["Low Volatility"] = - rolling_vol.values.flatten()
        
        if "Market" in sensi_factors:
            market_prices = universe_filtered['Market'].set_index("Date").reindex(price_df.index).ffill()
            market_return = market_prices.pct_change().dropna()

            beta_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
            for ticker in returns_df.columns:
                # Covariance entre l'actif et le marché sur la fenêtre glissante
                cov_series = returns_df[ticker].rolling(window=252).cov(market_return)
                # Variance du marché sur la même fenêtre
                var_series = market_return.rolling(window=252).var()
                # Calcul du beta pour chaque date de la fenêtre (résultat NaN pour les 251 premiers jours)
                beta_df[ticker] = cov_series / var_series

            beta_df_full = beta_df.reindex(price_df.index)

            factor_df["Market"] = beta_df_full.values.flatten()

        #  Filtrer sur les dates > start_date
        factor_df = factor_df.loc[factor_df["Date"] > computation_date_dt, :].reset_index(drop=True)
        
        # print("Calcul des facteurs terminé.")
        return factor_df

    def get_tickers_in_range(self, start_date_dt: datetime, end_date_dt: datetime = None) -> list:
        """
        Récupère tous les tickers présents dans l'indice entre deux dates

        Args:
            start_date (str): Date de début (format YYYY-MM-DD)
            end_date (str, optional): Date de fin (format YYYY-MM-DD). Default: None.

        Returns:
            list: Liste des tickers valides avec suffixe " Equity"
        """
        compo_universe = self.universe_data['compo']

        # Dernière compo avant start_date
        closest_row_universe = compo_universe[compo_universe['Date'] <= start_date_dt].sort_values('Date').tail(1)

        if closest_row_universe.empty:
            raise ValueError("Impossible de récupérer la composition de l'univers avant la start_date.")

        tickers_universe = set(
            ticker + " Equity" for ticker in closest_row_universe.loc[:, closest_row_universe.eq(1).iloc[0]].columns.to_list()
        )
        if end_date_dt:
            filtered_universe = compo_universe[(compo_universe['Date'] >= start_date_dt) & 
                                            (compo_universe['Date'] <= end_date_dt)]
            
            if not filtered_universe.empty:
                active_tickers = set(
                    ticker + " Equity" for ticker in filtered_universe.loc[:, filtered_universe.eq(1).any()].columns.to_list()
                )
                tickers_universe.update(active_tickers)

        # Vérifier les tickers en commun
        common_tickers = self.get_common_tickers(self.universe_data, list(tickers_universe))
        # print(f"Nombre de tickers sélectionnés : {len(tickers_universe)}")
        # print(f"Nombre de tickers communs dans toutes les données : {len(common_tickers)}")
        return common_tickers


    @staticmethod
    def get_common_tickers(dictionnary : dict, tickers_universe : list) -> set:
        dictionnary['Price'] = dictionnary['Price'].dropna(axis=1) 
        common_tickers = set(tickers_universe)
        return common_tickers.intersection(*(set(df.columns) for key, df in dictionnary.items() if key != 'compo'))

    @staticmethod
    def get_analysis_dates(price_df: pd.DataFrame, computation_date: datetime, 
                           n_before: int = 253, n_after: int = 252 * 3,
                           end_date: datetime = None):
        """
        Récupère les dates de début et de fin pour l'analyse en fonction des jours de marché.

        Args:
            price_df (pd.DataFrame): DataFrame contenant les prix avec une colonne 'Date'.
            computation_date (datetime): Date de référence pour l'analyse.
            n_before (int, optional): Nombre de jours de marché avant la computation_date. Par défaut 253.
            n_after (int, optional): Nombre de jours de marché après la computation_date. Par défaut 756 (252 * 3).
            end_date (datetime, optional): Date de fin spécifique. Si None, utilise n_after.
        Returns:
            tuple(datetime, datetime): (start_date, end_date)
        """
        price_dates = price_df['Date'].sort_values().unique()

        # Trouver les dates avant computation_date
        past_dates = price_dates[price_dates < computation_date]
        if len(past_dates) < n_before:
            raise ValueError(f"Pas assez de données avant {computation_date}. ({len(past_dates)} jours disponibles)")

        start_date = past_dates[-n_before]  # n_before jours avant

        if end_date:
            # Vérifier que end_date est dans le range
            if end_date <= computation_date:
                raise ValueError("end_date doit être postérieure à computation_date.")
            valid_future_dates = price_dates[price_dates > computation_date]
            if len(valid_future_dates) == 0 or end_date > valid_future_dates[-1]:
                raise ValueError("end_date dépasse les données disponibles.")

        else:
            # Utilisation de n_after si end_date n'est pas fourni
            future_dates = price_dates[price_dates > computation_date]
            if len(future_dates) < n_after:
                raise ValueError(f"Pas assez de données après {computation_date}. ({len(future_dates)} jours disponibles)")
            end_date = future_dates[n_after - 1]  # n_after jours après computation_date

        return start_date, end_date
    
    @staticmethod
    def calculate_transaction_costs(old_weights: dict, new_weights: dict, fees: float) -> float:
        """
        Calcule les frais de transaction basés sur les changements de poids.

        Args:
            old_weights (dict): Poids des actifs avant le rebalancement (ticker -> poids).
            new_weights (dict): Poids des actifs après le rebalancement (ticker -> poids).
            fees (float): Taux des frais de transaction (par exemple, 0.0005 pour 0.05%).

        Returns:
            float: Coût total des transactions.
        """
        # Obtenir l'ensemble des tickers impliqués
        all_tickers = set(old_weights.keys()).union(set(new_weights.keys()))

        # Calculer les frais de transaction pour chaque ticker
        transaction_costs = fees * np.sum(
            np.abs(np.array([new_weights.get(t, 0) - old_weights.get(t, 0) for t in all_tickers]))
        )

        return transaction_costs
        
    @staticmethod
    def compute_half_life_factor(factor_df, target_factor):
        # Trouver la première valeur du facteur cible
        initial_value, initial_date = factor_df[[target_factor, 'Date']].iloc[0]
        half_value = initial_value / 2
        half_life_date = None

        for date, value in zip(factor_df["Date"], factor_df[target_factor]):
            if value <= half_value:
                half_life_date = date
                break

        half_life_days = None
        if half_life_date:
            half_life_days = (half_life_date - initial_date).days
            # print(f"La demi-vie du facteur '{target_factor}' est atteinte à la date {half_life_date}")

            months_diff = (half_life_date.year - initial_date.year) * 12 + half_life_date.month - initial_date.month
            # print(f"Nombre de mois depuis la computation date: {months_diff} mois")
        else:
            # print(f"La valeur du facteur '{target_factor}' n'a jamais diminué de moitié dans la période analysée.")
            pass
        return half_life_date, half_life_days
    
    
    
    @staticmethod
    def plot_sensibilities(df : pd.DataFrame, facteur : pd.DataFrame, computation_date_str : str, half_life_date_dt : datetime = None):
        df["Date"] = pd.to_datetime(df["Date"])  
        start_date = df["Date"].iloc[0]  # Première date du dataset

        # Calcul du nombre de mois écoulés depuis la computation_date
        computation_date_dt = datetime.strptime(computation_date_str, "%Y-%m-%d")
        df["Absolute Undesired Exposure"] = df.drop(columns=["Date", facteur], errors="ignore").abs().sum(axis=1)
        df["Months_Since_Start"] = (df["Date"] - start_date) / pd.Timedelta(days=30)
        
        # Calcul du nombre de mois depuis la demi-vie (si elle existe)
        months_since_half_life = None
        if half_life_date_dt:
            months_since_half_life = (half_life_date_dt.year - computation_date_dt.year) * 12 + half_life_date_dt.month - computation_date_dt.month
        
        title = f"Évolution des facteurs avec une stratégie {facteur} composé au {computation_date_str}"
        if months_since_half_life is not None:
            title += f"<br>Nombre de mois depuis la demi-vie : {months_since_half_life} mois"
        
        fig = px.line(df, x="Months_Since_Start", y=df.columns[1:-1], 
                      labels={"value": "Facteur", "Months_Since_Start": "Holding Months Ahead"},
                      title=title)
        fig.show()


