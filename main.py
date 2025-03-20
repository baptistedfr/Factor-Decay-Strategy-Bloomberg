import pandas as pd
import warnings
from src.portfolio import FractilePortfolio, PureFactorPortfolio
from src.enums import Universe
from src.analysis import PortfolioAnalysis
import plotly.graph_objects as go
from tqdm import tqdm
warnings.filterwarnings("ignore")



def generate_half_life_analysis():
    ptf_analysis = PortfolioAnalysis(universe = Universe.SP500)
    sensi_factors = ["Momentum","Value","Quality","Low Volatility","Market"]
    portfolios = [FractilePortfolio, PureFactorPortfolio]
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

# generate_half_life_analysis()
ptf_analysis = PortfolioAnalysis(universe = Universe.SP500)
sensi_factors = ["Momentum","Value","Quality","Low Volatility","Market"]
sensis = ["Momentum"]
portfolio = PureFactorPortfolio


# date = "2014-06-30"


# sensibilities, half_life = ptf_analysis.get_factor_information(
#                         target_factor=sensi,
#                         sensi_factors=sensi_factors,
#                         computation_date_str=date,
#                         Portfolio=portfolio,
#                         plot = True  
#                     )

start_date = "2013-01-31"
end_date = "2018-01-31"
for sensi in sensis:
    combined_results = ptf_analysis.compare_strategies(
                            target_factor=sensi,
                            sensi_factors=sensi_factors,
                            start_date_str=start_date,
                            end_date_str=end_date,
                            Portfolio=portfolio,
                            transaction_fees = 0.0005
                        )

    print(combined_results.df_statistics.head(10))
    combined_results.df_statistics.to_excel(f"Results/{portfolio.__name__}_Strateg{sensi}_{start_date}_{end_date}.xlsx", index=True)
    combined_results.ptf_value_plot.show()
    combined_results.ptf_drawdown_plot.show() 

    for rebalancing_plot in combined_results.plt_rebalancing_plot:
        rebalancing_plot.show()


