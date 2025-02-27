import pandas as pd
import numpy as np
from portfolio import FractilePortfolio
from Utilities import Universe
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import plotly.express as px

logging.basicConfig(level=logging.INFO)

class PortfolioAnalysis:

    def __init__(self, universe : Universe):
        """
        Parameters:
        ------------
        universe : Universe
        """

        # Vérifier que les fichiers existent avant de les charger
        
        # logging.info("Chargement de l'univers...")
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

        # logging.info("Données chargées et triées par date.")
    

    def get_factor_information(self, target_factor: str, sensi_factors: list[str], computation_date_str : str, plot = True):
        """
        Construct the portfolio at the first date and get the sensi at each following date.
        Args:
            target_factor (str): Type of long-short portfolio to create (Momentum, Value, Growth ...)
            sensi_factors (list[str], optional): List of factors for which a sensitivity is calculated
            computation_date (str): Date to compute the analysis
        """
        computation_date_dt = datetime.strptime(computation_date_str, "%Y-%m-%d")

        compo_universe = self.universe_data['compo']
        closest_row_universe = compo_universe[compo_universe['Date'] <= computation_date_dt].sort_values('Date').tail(1)

        if closest_row_universe.empty:
            raise ValueError("Impossible de récupérer la composition de l'univers.")

        tickers_universe = [
            ticker + " Equity" for ticker in closest_row_universe.loc[:, closest_row_universe.eq(1).iloc[0]].columns.to_list()
        ]
        print(f"Nombre de tickers sélectionnés : {len(tickers_universe)}")
        common_tickers = self.get_common_tickers(self.universe_data, tickers_universe)
        print(f"Nombre de tickers communs dans toutes les données : {len(common_tickers)}")

        # Filtrage des données en fonction de la date et des tickers sélectionnés
        price_df = self.universe_data['Price']  # DataFrame Price
        start_date, end_date = self.get_analysis_dates(price_df, computation_date_dt)
        print(f"Période d'analyse : {start_date} -> {end_date}")
        
        self.universe_filtered = {
            key: df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), ['Date'] + list(common_tickers)]
            for key, df in self.universe_data.items() if key != 'compo'
        }
        price_df = self.universe_data['Price']
        market_df = price_df.loc[(price_df['Date'] >= start_date) & (price_df['Date'] <= end_date), ['Date', 'SPX Index']]
        self.universe_filtered['Market'] = market_df
        
        factor_df = self.__construct_dataframe_factors(sensi_factors, computation_date_dt)
        all_dates = sorted(set(factor_df["Date"].tolist()))

        ptf_construction = FractilePortfolio(df_factor=factor_df, target_factor=target_factor, sensi_factors=sensi_factors)

        initial_ptf, first_sensi = ptf_construction.process_ptf(factor_df.loc[factor_df["Date"] == all_dates[0], :], rebalance_weight=True)
        # Keep in memory the weights optimized for the given factor
        weights = initial_ptf["Weight"].tolist()

        # Compute portfolio sensibility to sensi factors for all dates
        sensi_records = [{"Date": factor_df['Date'].iloc[0], **first_sensi.to_dict()}]
        for date in all_dates[1:]:
            # Add the initial weights to the dataframe at the current date
            df_date = factor_df[factor_df["Date"] == date].copy()
            df_date["Weight"] = weights
            # Compute the sensi
            _, sensi_date = ptf_construction.process_ptf(df_date, rebalance_weight=False)
            sensi_records.append({"Date": date, **sensi_date.to_dict()})


        sensi_historic = pd.DataFrame(sensi_records)
        half_life_date, half_life_days = self.compute_half_life_factor(sensi_historic, target_factor) 

        if plot:
            self.plot_sensibilities(sensi_historic, target_factor, computation_date_str, half_life_date)

        
        return sensi_historic, half_life_days
    
    def __construct_dataframe_factors(self, sensi_factors: list[str], computation_date_dt : datetime):
        """
        Calcule les facteurs demandés et retourne un DataFrame avec une ligne par ticker/date.
        
        Args:
            sensi_factors (list[str]): Liste des facteurs à calculer.
        
        Returns:
            pd.DataFrame: DataFrame structuré avec une colonne par facteur.
        """
        # Récupérer les prix et aligner les autres datasets
        price_df = self.universe_filtered['Price'].set_index("Date")
        
        # Initialiser la structure du DataFrame final
        factor_df = pd.DataFrame()
        factor_df["Date"] = np.repeat(price_df.index, len(price_df.columns))
        factor_df["Ticker"] = np.tile(price_df.columns, len(price_df))

        # Ajouter chaque facteur comme une colonne
        if "Value" in sensi_factors:
            # print("Calcul du facteur Value (P/B)...")
            ptb_df = self.universe_filtered['Price To Book'].set_index("Date").reindex(price_df.index).ffill()
            factor_df["Value"] = ptb_df.values.flatten()

        if "Momentum" in sensi_factors:
            # print("Calcul du facteur Momentum (12-1 month return)...")
            P_t_12 = price_df.shift(252)
            P_t_1 = price_df.shift(21)
            factor_df["Momentum"] = (P_t_1 / P_t_12 - 1).values.flatten()

        if "Quality" in sensi_factors:
            # print("Calcul du facteur Quality (ROE)...")
            roe_df = self.universe_filtered['ROE'].set_index("Date").reindex(price_df.index).ffill()
            factor_df["Quality"] = roe_df.values.flatten()

        if "Low Volatility" in sensi_factors:
            # print("Calcul du facteur Low Volatility (252-day vol)...")
            rolling_vol = (price_df.pct_change().rolling(window=252).std())*np.sqrt(252)
            factor_df["Low Volatility"] = rolling_vol.values.flatten()
        
        if "Market" in sensi_factors:
            market_prices = self.universe_filtered['Market'].set_index("Date").reindex(price_df.index).ffill()
            returns_df = price_df.pct_change().dropna()
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
        factor_df = factor_df.loc[factor_df["Date"] > computation_date_dt, :]
        print("Calcul des facteurs terminé.")
        return factor_df

    @staticmethod
    def get_common_tickers(dictionnary : dict, tickers_universe : list) -> set:
        dictionnary['Price'] = dictionnary['Price'].dropna(axis=1) 
        common_tickers = set(tickers_universe)
        return common_tickers.intersection(*(set(df.columns) for key, df in dictionnary.items() if key != 'compo'))

    @staticmethod
    def get_analysis_dates(price_df: pd.DataFrame, computation_date: datetime, n_before: int = 253, n_after: int = 252 * 3):
        """
        Récupère les dates de début et de fin pour l'analyse en fonction des jours de marché.

        Args:
            price_df (pd.DataFrame): DataFrame contenant les prix avec une colonne 'Date'.
            computation_date (datetime): Date de référence pour l'analyse.
            n_before (int, optional): Nombre de jours de marché avant la computation_date. Par défaut 253.
            n_after (int, optional): Nombre de jours de marché après la computation_date. Par défaut 756 (252 * 3).

        Returns:
            tuple(datetime, datetime): (start_date, end_date)
        """
        # Trier et extraire les dates uniques du dataset Price
        price_dates = price_df['Date'].sort_values().unique()

        # Trouver les dates avant computation_date
        past_dates = price_dates[price_dates < computation_date]
        if len(past_dates) < n_before:
            raise ValueError(f"Pas assez de données avant {computation_date}. ({len(past_dates)} jours disponibles)")

        start_date = past_dates[-n_before]  # n_before jours avant

        # Trouver les dates après computation_date
        future_dates = price_dates[price_dates > computation_date]
        if len(future_dates) < n_after:
            raise ValueError(f"Pas assez de données après {computation_date}. ({len(future_dates)} jours disponibles)")

        end_date = future_dates[n_after - 1]  # n_after jours après

        return start_date, end_date
    
    @staticmethod
    def compute_half_life_factor(factor_df, target_factor):
        # Trouver la première valeur du facteur cible
        initial_value, initial_date = factor_df[[target_factor, 'Date']].iloc[0]

        # Trouver la première date où la valeur est divisée par 2
        half_value = initial_value / 2
        half_life_date = None

        for date, value in zip(factor_df["Date"], factor_df[target_factor]):
            if value <= half_value:
                half_life_date = date
                break

        half_life_days = None
        if half_life_date:
            half_life_days = (half_life_date - initial_date).days
            print(f"La demi-vie du facteur '{target_factor}' est atteinte à la date {half_life_date}")

            # Calcul du nombre de mois depuis la computation_date
            months_diff = (half_life_date.year - initial_date.year) * 12 + half_life_date.month - initial_date.month
            print(f"Nombre de mois depuis la computation date: {months_diff} mois")
        else:
            print(f"La valeur du facteur '{target_factor}' n'a jamais diminué de moitié dans la période analysée.")

        return half_life_date, half_life_days
    
    
    
    @staticmethod
    def plot_sensibilities(df, facteur : pd.DataFrame, computation_date_str : str, half_life_date_dt : datetime = None):
        df["Date"] = pd.to_datetime(df["Date"])  
        start_date = df["Date"].iloc[0]  # Première date du dataset

        # Calcul du nombre de mois écoulés depuis la computation_date
        computation_date_dt = datetime.strptime(computation_date_str, "%Y-%m-%d")
        df["Months_Since_Start"] = (df["Date"] - start_date) / pd.Timedelta(days=30)
        
        # Calcul du nombre de mois depuis la demi-vie (si elle existe)
        months_since_half_life = None
        if half_life_date_dt:
            months_since_half_life = (half_life_date_dt.year - computation_date_dt.year) * 12 + half_life_date_dt.month - computation_date_dt.month
        
        # Construction du titre
        title = f"Évolution des facteurs avec une stratégie {facteur} composé au {computation_date_str}"
        if months_since_half_life is not None:
            title += f"<br>Nombre de mois depuis la demi-vie : {months_since_half_life} mois"
        
        # Création du graphique interactif avec Plotly
        fig = px.line(df, x="Months_Since_Start", y=df.columns[1:-1], 
                      labels={"value": "Facteur", "Months_Since_Start": "Holding Months Ahead"},
                      title=title)

        # Affichage du graphique
        fig.show()


