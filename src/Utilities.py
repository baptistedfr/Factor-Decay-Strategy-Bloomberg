import pandas as pd
import warnings
import plotly.graph_objects as go
from tqdm import tqdm
from enums import FrequencyType

def get_rebalancing_dates(dates : list, frequency: FrequencyType) -> list[int]:
        """
        Repère les indices correspondant aux dates de rebalancement en fonction de la fréquence donnée.
        
        Args:
         dates (list[pd.Timestamp]): Liste des dates disponibles.
            frequency (FrequencyType): Fréquence de rebalancement souhaitée.
        
        Returns:
            list[int]: Indices des dates de rebalancement.
        """
        date_series = pd.Series(dates).sort_values().reset_index(drop=True)
        if frequency == FrequencyType.MONTHLY:
            # On récupère la dernière date de chaque mois
            rebalancing_dates = date_series.groupby(date_series.dt.to_period("M")).last().tolist()
        elif frequency == FrequencyType.WEEKLY:
            # On récupère la dernière date de chaque semaine
            rebalancing_dates = date_series.groupby(date_series.dt.to_period("W")).last().tolist()
        elif frequency == FrequencyType.DAILY:
            # Toutes les dates sont des dates de rebalancement
            rebalancing_dates = date_series.tolist()
        else:
            raise ValueError("Fréquence non reconnue. Utilisez 'MONTHLY', 'WEEKLY' ou 'DAILY'.")
        
        indices = [date_series[date_series == d].index[0] for d in rebalancing_dates]

        return indices