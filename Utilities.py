from enum import Enum
import pandas as pd
import warnings
from portfolio import FractilePortfolio, PureFactorPortfolio
import plotly.graph_objects as go
from tqdm import tqdm
from analysis import PortfolioAnalysis, Universe

def generate_half_life_analysis():
    ptf_analysis = PortfolioAnalysis(universe = Universe.SP500)
    sensi_factors = ["Momentum","Value","Quality","Low Volatility","Market"]
    portfolios = [PureFactorPortfolio, FractilePortfolio]
    dates = pd.date_range(start="2004-01-31", end="2025-01-31", freq="BM")

    for ptf_type in portfolios:
        df = pd.DataFrame(columns = sensi_factors)
        for i in tqdm(range(len(dates)), desc = ptf_type.__name__ + " Computation"):
            for sensi in sensi_factors:
                try:
                    sensibilities, half_life = ptf_analysis.get_factor_information(
                        target_factor=sensi,
                        sensi_factors=sensi_factors,
                        computation_date_str=dates[i].strftime("%Y-%m-%d"),
                        Portfolio=ptf_type,
                        plot = False  
                    )
                    df.loc[dates[i], sensi] = half_life
                except Exception as e:
                    print(f"⚠️ Erreur sur {ptf_type.__name__}, facteur {sensi}, date {dates[i]}: {e}")
                    df.loc[dates[i], sensi] = None   

        # Sauvegarde du fichier Excel avec le bon nom
        df.to_excel(f"{ptf_type.__name__}.xlsx", index=True)
        print(f"✅ Sauvegarde de {ptf_type.__name__}.xlsx terminée.")