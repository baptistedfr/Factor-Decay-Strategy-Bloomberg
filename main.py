import pandas as pd
import warnings
from portfolio import FractilePortfolio
import plotly.graph_objects as go
from Utilities import Universe
warnings.filterwarnings("ignore")


# df_factors = pd.read_excel("Input/S&P500_Factors.xlsx")
# df_factors = df_factors.rename(columns={"Value (P/B)": "Value", 'Momentum (12m)': "Momentum",
#                                         "Low Volatility (252d)": "Low Vol", 'Quality (ROE)': "Quality"})

from analysis import PortfolioAnalysis
ptf_analysis = PortfolioAnalysis(universe = Universe.SP500)
sensibilities = ptf_analysis.get_factor_information("Quality",["Momentum","Value","Quality","Low Volatility"],"2017-01-01")
