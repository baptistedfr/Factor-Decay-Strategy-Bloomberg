import pandas as pd
import warnings
from portfolio import FractilePortfolio, PureFactorPortfolio
import plotly.graph_objects as go
from tqdm import tqdm
from Utilities import generate_half_life_analysis 
warnings.filterwarnings("ignore")


from analysis import PortfolioAnalysis, Universe



# generate_half_life_analysis()
ptf_analysis = PortfolioAnalysis(universe = Universe.SP500)
sensi_factors = ["Momentum","Value","Quality","Low Volatility","Market"]
sensi = "Momentum"
portfolio = PureFactorPortfolio
date = "2004-03-31"

sensibilities, half_life = ptf_analysis.get_factor_information(
                        target_factor=sensi,
                        sensi_factors=sensi_factors,
                        computation_date_str=date,
                        Portfolio=portfolio,
                        plot = True  
                    )