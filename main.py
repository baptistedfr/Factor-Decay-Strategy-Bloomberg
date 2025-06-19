import pandas as pd
import warnings
from src.portfolio import FractilePortfolio, PureFactorPortfolio
from src.enums import Universe
from src.analysis import PortfolioAnalysis
import plotly.graph_objects as go
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed


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



def run_factor_backtest(
    ptf_analysis,
    portfolio_class,
    sensi_factors,
    sensis,
    global_start="2004-01-31",
    global_end="2018-01-31",
    window_years=5,
    frequency='Q',
    output_path="Results/FilteredStats_5y_byQuarter.xlsx",
    verbose=False
):
    start_global = pd.Timestamp(global_start)
    end_global = pd.Timestamp(global_end)

    all_results = {sensi: [] for sensi in sensis}
    start_dates = pd.date_range(start=start_global, end=end_global, freq=frequency)

    #for start_date in tqdm(start_dates, desc=f"Backtests de {portfolio_class.__name__} par {frequency}"):
    for start_date in start_dates:
        print(start_date)
        end_date = start_date + relativedelta(years=window_years)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        for sensi in sensis:
            try:
                combined_results = ptf_analysis.compare_strategies(
                    target_factor=sensi,
                    sensi_factors=sensi_factors,
                    start_date_str=start_str,
                    end_date_str=end_str,
                    Portfolio=portfolio_class,
                    transaction_fees=0
                )
                metrics_to_keep = ["Total Return", "Annualized Return", "Volatility"]
                df_stats = combined_results.df_statistics.copy()
                df_stats = df_stats.set_index(df_stats.columns[0])  # Mets "Metrics" en index
                df_stats_filtered = df_stats.loc[metrics_to_keep]

                # Aplatit : une seule ligne avec chaque (metric, stratégie) comme colonne
                df_t = df_stats_filtered.T
                columns = []
                values = []

                for strategy_name, row in df_t.iterrows():
                    for metric in metrics_to_keep:
                        columns.append(f"{metric} - {strategy_name}")
                        values.append(row[metric])

                df_flat = pd.DataFrame([values], columns=columns)
                df_flat.insert(0, "Start Date", start_str)
                df_flat.insert(1, "End Date", end_str)

                all_results[sensi].append(df_flat)

            except Exception as e:
                if verbose:
                    print(f"Erreur pour {sensi} du {start_str} au {end_str} : {e}")

    # Sauvegarde finale
    with pd.ExcelWriter(output_path) as writer:
        for sensi, df_list in all_results.items():
            if df_list:
                df_concat = pd.concat(df_list, ignore_index=True)
                df_concat.to_excel(writer, sheet_name=sensi, index=False)

    if verbose:
        print(f"Résultats filtrés exportés dans : {output_path}")


# generate_half_life_analysis()
# ptf_analysis = PortfolioAnalysis(universe = Universe.SP500)
# sensi_factors = ["Momentum","Value","Quality","Low Volatility","Market"]
# sensis = ["Momentum","Value","Quality","Low Volatility"]
# portfolio = PureFactorPortfolio


# date = "2014-06-30"


# sensibilities, half_life = ptf_analysis.get_factor_information(
#                         target_factor=sensi,
#                         sensi_factors=sensi_factors,
#                         computation_date_str=date,
#                         Portfolio=portfolio,
#                         plot = True  
#                     )

# start_date = "2013-01-31"
# end_date = "2018-01-31"
# ptf_analysis = PortfolioAnalysis(universe = Universe.SP500)
# sensi_factors = ["Momentum","Value","Quality","Low Volatility","Market"]
# sensis = ["Momentum","Value","Quality","Low Volatility"]
# portfolio = PureFactorPortfolio
# for sensi in sensis:
#     combined_results = ptf_analysis.compare_strategies(
#                             target_factor=sensi,
#                             sensi_factors=sensi_factors,
#                             start_date_str=start_date,
#                             end_date_str=end_date,
#                             Portfolio=portfolio,
#                             transaction_fees = 0
#                         )

#     print(combined_results.df_statistics.head(10))
#     combined_results.df_statistics.to_excel(f"Results/{portfolio.__name__}_Strateg{sensi}_{start_date}_{end_date}.xlsx", index=True)
#     combined_results.ptf_value_plot.show()
#     combined_results.ptf_drawdown_plot.show() 

#     for rebalancing_plot in combined_results.plt_rebalancing_plot:
#         rebalancing_plot.show()



ptf_analysis = PortfolioAnalysis(universe=Universe.SP500)
start_date = "2013-01-31"
end_date = "2018-01-31"
ptf_analysis = PortfolioAnalysis(universe = Universe.SP500)
sensi_factors = ["Momentum","Value","Quality","Low Volatility","Market"]
sensis = ["Momentum","Value","Quality","Low Volatility"]
sensi = "Momentum"
portfolio = PureFactorPortfolio
combined_results, liste = ptf_analysis.compare_strategies(
                            target_factor=sensi,
                            sensi_factors=sensi_factors,
                            start_date_str=start_date,
                            end_date_str=end_date,
                            Portfolio=portfolio,
                            transaction_fees = 0
                        )

print(combined_results.df_statistics.head(10))
combined_results.df_statistics.to_excel(f"Results/{portfolio.__name__}_Strateg{sensi}_{start_date}_{end_date}.xlsx", index=True)
combined_results.ptf_value_plot.show()
combined_results.ptf_drawdown_plot.show() 

for rebalancing_plot in combined_results.plt_rebalancing_plot:
    rebalancing_plot.show()
# run_factor_backtest(
#     ptf_analysis=ptf_analysis,
#     portfolio_class=PureFactorPortfolio,
#     sensi_factors=["Momentum", "Value", "Quality", "Low Volatility", "Market"],
#     sensis=["Momentum", "Value", "Quality", "Low Volatility"],
#     global_start = "2004-01-31",
#     global_end = "2019-04-30",
#     window_years = 5,
#     frequency= 'Q',
#     output_path = "Results/PureFactorPortfolio_5Y_Quarterly_Stats.xlsx",
#     verbose=True
# )

# run_factor_backtest(
#     ptf_analysis=ptf_analysis,
#     portfolio_class=FractilePortfolio,
#     sensi_factors=["Momentum", "Value", "Quality", "Low Volatility", "Market"],
#     sensis=["Momentum", "Value", "Quality", "Low Volatility"],
#     global_start = "2004-01-31",
#     global_end = "2019-04-30",
#     window_years = 5,
#     frequency= 'Q',
#     output_path = "Results/FractilePortfolio_5Y_Quarterly_Stats.xlsx",
#     verbose=True
# )