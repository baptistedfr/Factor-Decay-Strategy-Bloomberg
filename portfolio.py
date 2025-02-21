from typing import Tuple
import pandas as pd
from scipy.stats import zscore
from enum import Enum

class Weighting_Type(Enum):
    EQUAL_WEIGHT = "EW"


class FractilePortfolio:

    def __init__(self, df_factor: pd.DataFrame, target_factor: str, sensi_factors: list[str] = None,
                 nb_fractile: int = 4, weighting_type: Weighting_Type = Weighting_Type.EQUAL_WEIGHT):
        """
        Parameters:
        ------------
        df_universe : pd.DataFrame
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
        self.weighting_type = weighting_type
        self.target_factor = target_factor
        self.df_factor = df_factor
        self.nb_fractile = nb_fractile

        if self.nb_fractile < 2:
            raise ValueError("Le nombre de fractiles doit être au moins 2.")

        cols = self.df_factor.columns
        if not all(factor in cols for factor in sensi_factors) or "Date" not in cols or "Ticker" not in cols:
            raise ValueError("Certaines colonnes obligatoires ['Date', 'Ticker'] ou des facteurs de sensibilité sont absentes du DataFrame.")

        self.sensi_factors = sensi_factors if sensi_factors else [x for x in self.df_factor.columns if x not in ["Date", "Ticker"]]


    def _compute_zscore_fractile(self, df):
        """
        Compute the Zscore for each factor and cut the universe into fractiles.
        """
        for factor in self.sensi_factors:
            zscore_col = f'Zscore_{factor}'
            quartile_col = f'Fractile_{factor}'

            df[zscore_col] = zscore(df[factor], nan_policy='omit')
            df[zscore_col] = df[zscore_col].fillna(0)
            df[quartile_col] = pd.qcut(df[zscore_col], q=self.nb_fractile,
                                                     labels=[x+1 for x in range(self.nb_fractile)])
        return df

    def _apply_weights(self, df_ptf : pd.DataFrame) -> pd.DataFrame:
        """
        Create the long-short portfolio by shorting the last fractile and buying the first.
        Apply the selected weighting scheme.

        Returns:
        ----------
        pd.DataFrame : titles in portfolio
        """

        num_long = len(df_ptf[df_ptf[f"Fractile_{self.target_factor}"] == self.nb_fractile])
        num_short = len(df_ptf[df_ptf[f"Fractile_{self.target_factor}"] == 1])

        if num_long == 0 or num_short == 0:
            raise ValueError("Un des fractiles est vide, impossible d'appliquer les poids.")

        # The pd.cut function compute the 1 fractile as the worst fractile
        if self.weighting_type == Weighting_Type.EQUAL_WEIGHT:
            df_ptf['Weight'] = df_ptf[f"Fractile_{self.target_factor}"].apply(
                lambda x: 1 / num_long if x == self.nb_fractile else 
                        -1 / num_short if x == 1 else 
                        0)
        else:
            raise Exception("Weighting method not implemented yet or not existing")

        return df_ptf

    def _compute_factor_exposure(self, df_ptf: pd.DataFrame) -> pd.Series:
        """
        Compute the exposure to each sensi_factor in the portfolio dataframe

        Parameters:
        ------------
        df_ptf : pd.DataFrame
            Titles in the portfolio

        Returns:
        ----------
        pd.Series : sensibilities of the portfolio
        """
        factor_exposures = {}
        for factor in self.sensi_factors:
            factor_exposures[factor] = (df_ptf['Weight'] * df_ptf[f'Zscore_{factor}']).sum()

        return pd.Series(factor_exposures)
        #return (df_ptf.set_index('Ticker')[self.sensi_factors].mul(df_ptf['Weight'], axis=0)).sum()

    def process_ptf(self, df_factor : pd.DataFrame, rebalance_weight: bool = False, save: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create the portfolio with maximal target factor exposure.
        Compute the portfolio sensibility to each sensi factors.

        Parameters:
        ------------
        save : bool
            Save or not the portfolio in csv format
        """
        df_ptf = self._compute_zscore_fractile(df_factor)
        if rebalance_weight:
            df_ptf = self._apply_weights(df_ptf)
        else:
            if 'Weight' not in df_ptf.columns:
                raise ValueError("Please Provide a Weight column")

        ptf_sensi = self._compute_factor_exposure(df_ptf)

        if save:
            df_ptf.to_csv(f"Output//Portfolio_{self.target_factor}_{self.nb_fractile}F.csv", index=False)
            ptf_sensi.to_csv(f"Output//Sensi_{self.target_factor}_{self.nb_fractile}F.csv", index=False)

        return df_ptf, ptf_sensi 

