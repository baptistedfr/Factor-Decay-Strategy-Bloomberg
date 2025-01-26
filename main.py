import pandas as pd
import warnings
from portfolio import FractilePortfolio
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

# def fetch_equity_factors(tickers):
#     data = []
#     for ticker in tickers:
#         stock = yf.Ticker(ticker)
#
#         try:
#             # Price-to-Book Ratio (Value)
#             pb_ratio = stock.info.get('priceToBook', None)
#
#             # 12-month Momentum (Momentum)
#             hist = stock.history(period='1y')
#             if not hist.empty:
#                 momentum = (hist['Close'][-1] / hist['Close'][0]) - 1
#             else:
#                 momentum = None
#
#             # Return on Equity (Quality)
#             roe = stock.info.get('returnOnEquity', None)
#
#             # Percentage Growth in Total Assets (Investment)
#             total_assets = stock.balance_sheet.loc['Total Assets'] if 'Total Assets' in stock.balance_sheet else None
#             if total_assets is not None and len(total_assets) > 1:
#                 investment = (total_assets[-1] - total_assets[-2]) / total_assets[-2]
#             else:
#                 investment = None
#
#             # 252-day return volatility (Low Volatility)
#             if not hist.empty:
#                 daily_returns = hist['Close'].pct_change().dropna()
#                 low_volatility = daily_returns.std() * (252 ** 0.5)
#             else:
#                 low_volatility = None
#
#             # Append to data
#             data.append({
#                 'Ticker': ticker,
#                 'Value (P/B)': pb_ratio,
#                 'Momentum': momentum,
#                 'Quality (ROE)': roe,
#                 'Investment (Asset Growth)': investment,
#                 'Low Volatility (252d)': low_volatility,
#             })
#
#         except Exception as e:
#             print(f"Error fetching data for {ticker}: {e}")
#
#     return pd.DataFrame(data)
#
# df = pd.read_json("sp500.json")
# tickers = df["Ticker"].tolist()
# df_infos = fetch_equity_factors(tickers)
# print(df_infos.head())
# df_infos.to_csv("sp500_factors.csv", index=False)

df_factors = pd.read_excel("Input/S&P500_Factors.xlsx")
df_factors = df_factors.rename(columns={"Value (P/B)": "Value", 'Momentum (12m)': "Momentum",
                                        "Low Volatility (252d)": "Low Vol", 'Quality (ROE)': "Quality"})
dates = set(df_factors["Date"])

df_sensi = pd.DataFrame()
for date in sorted(dates):
    df_date = df_factors[df_factors["Date"] == date]
    ptf = FractilePortfolio(df_universe=df_date, target_factor="Momentum")
    df_ptf, ptf_sensi = ptf.process_ptf(save=False)
    df_sensi = pd.concat([df_sensi, pd.DataFrame([{"Date": date, **ptf_sensi}])])

print(df_sensi)

df_sensi["Date"] = pd.to_datetime(df_sensi["Date"])
fig = go.Figure()
for column in df_sensi.columns[1:]:
    fig.add_trace(go.Scatter(
        x=df_sensi["Date"],
        y=df_sensi[column],
        mode="lines+markers",
        name=column
    ))
fig.update_layout(
    title="Évolution des facteurs au fil du temps",
    xaxis_title="Date",
    yaxis_title="Valeur",
    legend_title="Facteur",
    template="plotly_white"
)
fig.show()
