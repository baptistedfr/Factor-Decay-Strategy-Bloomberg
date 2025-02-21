import pandas as pd
import numpy as np
from portfolio import FractilePortfolio
from Utilities import Universe
from datetime import datetime
import logging
import plotly.express as px

logging.basicConfig(level=logging.INFO)

class PortfolioAnalysis:

    def __init__(self, universe : Universe):
        """
        Parameters:
        ------------
        universe : Universe

            Titles universe with the following columns : Date, Ticker & factor exposure for all 'sensi_factors'
        target_factor : str
            Type of long-short portfolio to create (Momentum, Value, Growth ...)
        sensi_factors : list[str]
            List of factors for which a sensitivity is calculated
        nb_fractile : int
            Number of fractile to cut the universe into (default = 4)
        weighting_type : enum Weighting_Type
            Type of weighting applied during the portfolio creation (default = Equal weights)
        """


        # Vérifier que les fichiers existent avant de les charger
        

        logging.info("Chargement de l'univers...")
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

        logging.info("Données chargées et triées par date.")
    

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
        # Définition des bornes de filtrage
        start_date = computation_date_dt - pd.DateOffset(months=12)  # 12 mois avant
        end_date = computation_date_dt + pd.DateOffset(months=36)    # 36 mois après

        self.universe_filtered = {
            key: df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), ['Date'] + list(common_tickers)]
            for key, df in self.universe_data.items() if key != 'compo'
        }

        factor_df = self.__construct_dataframe_factors(sensi_factors, computation_date_dt)
        all_dates = sorted(set(factor_df["Date"].tolist()))

        ptf_construction = FractilePortfolio(df_factor=factor_df, target_factor=target_factor, sensi_factors=sensi_factors)

        initial_ptf, first_sensi = ptf_construction.process_ptf(factor_df.loc[factor_df["Date"] == all_dates[0], :], rebalance_weight=True, save=False)
        # Keep in memory the weights optimized for the given factor
        weights = initial_ptf["Weight"].tolist()

        # Compute portfolio sensibility to sensi factors for all dates
        sensi_records = [{"Dates": factor_df['Date'].iloc[0], **first_sensi.to_dict()}]
        for date in all_dates[1:]:
            # Add the initial weights to the dataframe at the current date
            df_date = factor_df[factor_df["Date"] == date].copy()
            df_date["Weight"] = weights
            # Compute the sensi
            _, sensi_date = ptf_construction.process_ptf(df_date, rebalance_weight=False, save=False)
            sensi_records.append({"Dates": date, **sensi_date.to_dict()})

        sensi_historic = pd.DataFrame(sensi_records)
        if plot:
            self.plot_sensibilities(sensi_historic, target_factor)
        return sensi_historic
    
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
            print("Calcul du facteur Value (P/B)...")
            ptb_df = self.universe_filtered['Price To Book'].set_index("Date").reindex(price_df.index).ffill()
            factor_df["Value"] = ptb_df.values.flatten()

        if "Momentum" in sensi_factors:
            print("Calcul du facteur Momentum (12-1 month return)...")
            P_t_12 = price_df.shift(252)
            P_t_1 = price_df.shift(21)
            factor_df["Momentum"] = (P_t_1 / P_t_12 - 1).values.flatten()

        if "Quality" in sensi_factors:
            print("Calcul du facteur Quality (ROE)...")
            roe_df = self.universe_filtered['ROE'].set_index("Date").reindex(price_df.index).ffill()
            factor_df["Quality"] = roe_df.values.flatten()

        if "Low Volatility" in sensi_factors:
            print("Calcul du facteur Low Volatility (252-day vol)...")
            rolling_vol = (price_df.pct_change().rolling(window=252).std())*np.sqrt(252)
            factor_df["Low Volatility"] = rolling_vol.values.flatten()

        #  Filtrer sur les dates > start_date
        factor_df = factor_df.loc[factor_df["Date"] >= computation_date_dt, :]
        print("Calcul des facteurs terminé.")
        return factor_df

    @staticmethod
    def get_common_tickers(dictionnary, tickers_universe):
        common_tickers = set(tickers_universe)
        return common_tickers.intersection(*(set(df.columns) for key, df in dictionnary.items() if key != 'compo'))
    
    @staticmethod
    def plot_sensibilities(df, facteur):
        df["Dates"] = pd.to_datetime(df["Dates"])  
        start_date = df["Dates"].iloc[0]  # Première date du dataset

        # Calcul du nombre de mois écoulés
        df["Months_Since_Start"] = (df["Dates"] - start_date) / pd.Timedelta(days=30)

        # Création du graphique interactif avec Plotly
        fig = px.line(df, x="Months_Since_Start", y=df.columns[1:-1], 
                    labels={"value": "Facteur", "Months_Since_Start": "Holding Months Ahead"},
                    title=f"Évolution des sensibilités des facteurs dans le temps avec une stratégie {facteur}")

        # Affichage du graphique
        fig.show()

