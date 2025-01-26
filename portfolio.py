from typing import Tuple
import pandas as pd
from scipy.stats import zscore
from enum import Enum

class Weighting_Type(Enum):
    EQUAL_WEIGHT = "EW"


class FractilePortfolio:

    def __init__(self, df_universe: pd.DataFrame, target_factor: str, sensi_factors: list[str] = None,
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
        self.df_universe = df_universe.dropna().reset_index(drop=True)
        self.nb_fractile = nb_fractile
        if sensi_factors is None:
            self.sensi_factors = [x for x in df_universe.columns if x not in ["Date", "Ticker"]]
        else:
            self.sensi_factors = sensi_factors
            cols = df_universe.columns
            if all(sensi_factors) not in cols or "Date" not in cols or "Ticker" not in cols:
                raise Exception("Not all columns ['Date', 'Ticker] or sensi columns are in the source dataframe")

    def _compute_zscore_fractile(self):
        """
        Compute the Zscore for each factor and cut the universe into fractiles.
        """
        for factor in self.sensi_factors:
            zscore_col = f'Zscore {factor}'
            quartile_col = f'Fractile {factor}'

            self.df_universe[zscore_col] = zscore(self.df_universe[factor])
            self.df_universe[quartile_col] = pd.qcut(self.df_universe[zscore_col], q=self.nb_fractile,
                                                     labels=[x+1 for x in range(self.nb_fractile)])

    def _apply_weights(self) -> pd.DataFrame:
        """
        Create the long-short portfolio by shorting the last fractile and buying the first.
        Apply the selected weighting scheme.

        Returns:
        ----------
        pd.DataFrame : titles in portfolio
        """
        df_ptf = self.df_universe[self.df_universe[f"Fractile {self.target_factor}"].isin([1, self.nb_fractile])]

        num_long = len(df_ptf[df_ptf[f"Fractile {self.target_factor}"] == 1])
        num_short = len(df_ptf[df_ptf[f"Fractile {self.target_factor}"] == self.nb_fractile])

        # The pd.cut function compute the 1 fractile as the worst fractile
        if self.weighting_type == Weighting_Type.EQUAL_WEIGHT:
            df_ptf['Weight'] = df_ptf[f"Fractile {self.target_factor}"].apply(
                    lambda x: 1 / num_long if x == self.nb_fractile else -1 / num_short)
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
            factor_exposures[factor] = (df_ptf['Weight'] * df_ptf[f'Zscore {factor}']).sum()

        return pd.Series(factor_exposures)

    def process_ptf(self, save: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create the portfolio and compute the factor exposure.

        Parameters:
        ------------
        save : bool
            Save or not the data in csv format
        """
        self._compute_zscore_fractile()
        df_ptf = self._apply_weights()

        ptf_sensi = self._compute_factor_exposure(df_ptf)

        if save:
            df_ptf.to_csv(f"Output//Portfolio_{self.target_factor}_{self.nb_fractile}F.csv", index=False)
            ptf_sensi.to_csv(f"Output//Sensi_{self.target_factor}_{self.nb_fractile}F.csv", index=False)
        return df_ptf, ptf_sensi
