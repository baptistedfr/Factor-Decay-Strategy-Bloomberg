import pandas as pd
from portfolio import FractilePortfolio

class PortfolioAnalysis:

    def __init__(self, df_universe: pd.DataFrame, target_factor: str, sensi_factors: list[str] = None):
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
        self.target_factor = target_factor
        self.df_universe = df_universe.dropna().reset_index(drop=True)
        if sensi_factors is None:
            self.sensi_factors = [x for x in df_universe.columns if x not in ["Date", "Ticker"]]
        else:
            self.sensi_factors = sensi_factors
            cols = df_universe.columns
            if all(sensi_factors) not in cols or "Date" not in cols or "Ticker" not in cols:
                raise Exception("Not all columns ['Date', 'Ticker] or sensi columns are in the source dataframe")
    
    def process_all_dates(self):
        """
        Construct the portfolio at the first date and get the sensi at each following date.
        """

        all_dates = list(set(self.df_universe["Date"].tolist()))
        print(all_dates)

        # Build the portfolio with target factor at t0
        df_first_date = self.df_universe[self.df_universe["Date"] == all_dates[0]]
        ptf_construction = FractilePortfolio(df_universe=df_first_date, target_factor=self.target_factor)
        initial_ptf, first_sensi = ptf_construction.process_ptf(build=True, save=False)
        print(len(df_first_date))
        # Keep in memory the weights optimized for the given factor
        weights = initial_ptf["Weight"].tolist()
        print(len(weights))
        # Compute portfolio sensibility to sensi factors for all dates
        col_df = ["Dates"] + [x for x in self.sensi_factors]
        sensi_historic = pd.DataFrame(columns=col_df)
        sensi_historic = sensi_historic._append(pd.Series({"Dates": all_dates[0], **first_sensi}), ignore_index=True)
        for date in all_dates[1:]:
            # Add the initial weights to the dataframe at the current date
            df_date = self.df_universe[self.df_universe["Date"] == date]
            print(len(df_date))
            df_date["Weight"] = weights
            # Compute the sensi
            _, sensi_date = ptf_construction.process_ptf(build=False, save=False, df_new=df_date)
            sensi_historic = sensi_historic._append(pd.Series({"Dates": date, **sensi_date}), ignore_index=True)
