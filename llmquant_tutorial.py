import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import yfinance as yf

# # 读取 38 家科技股的历史数据
# tickers = ['AAPL', 'ACN', 'ADI', 'ADP', 'ADSK', 'ANSS', 'APH', 'BABA', 'BIDU', 'BR', 'CRM',
#            'FFIV', 'FIS', 'FISV','GOOG', 'GPN', 'IBM','INTC', 'INTU', 'IPGP', 'IT', 'JKHY', 
#            'KEYS', 'KLAC', 'LRCX', 'MA', 'MCHP', 'MSFT','MSI', 'NVDA', 'NXPI', 'PYPL', 'SNPS', 
#            'TEL', 'TTWO', 'TXN', 'V', 'VRSN']

tickers = ['AAPL','MSFT','NVDA','GOOG']

data = yf.download(tickers, '2018-1-1', '2024-3-1')['Close']

# 计算每日收益率
daily_stock_returns = (data - data.shift(1)) / data.shift(1)
daily_stock_returns.dropna(inplace=True)

# 根据前一天的收益率进行降序排名
df_rank = daily_stock_returns.rank(axis=1, ascending=False, method='min')

# 依据排名生成交易信号
df_signal = df_rank.copy()

for ticker in tickers:
    # 排名靠前的做空 ( -1 )，排名靠后的做多 ( +1 ) 卖涨买跌
    df_signal[ticker] = np.where(df_signal[ticker] < 2, -1, 1)

# breakpoint()
# 计算根据交易信号可能带来的下一日收益
returns = df_signal.mul(daily_stock_returns.shift(-1), axis=0)

# 对所有股票的收益做平均
strategy_returns = np.sum(returns, axis=1)/len(tickers)
print(strategy_returns)

if not strategy_returns.empty:
    # 累计收益率
    cumulative_returns = (strategy_returns + 1).cumprod()

    # 夏普比率 (假设无风险利率为 0)
    daily_rf_rate = 0
    annual_rf_rate = daily_rf_rate * 252
    strategy_volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = (strategy_returns.mean() - annual_rf_rate) / strategy_volatility

    # 最大回撤
    cum_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cum_max) / cum_max
    max_drawdown = drawdown.min()

    print("Cumulative Returns:")
    print(cumulative_returns[-1] if not cumulative_returns.empty else "No trades executed.")
    print("\nSharpe Ratio:")
    print(sharpe_ratio)
    print("\nMax Drawdown:")
    print(max_drawdown)
else:
    print("No trades executed. Cannot compute performance metrics.")

import matplotlib.pyplot as plt

# 绘制累计收益曲线
if not strategy_returns.empty:
    cumulative_returns = (strategy_returns + 1).cumprod()
    plt.figure(figsize=(10, 6))
    cumulative_returns.plot()
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.show()

    # 设定 6 个月（126 个交易日）的滚动窗口计算滚动夏普比率
    rolling_window = 126
    rolling_sharpe_ratio = (strategy_returns.rolling(window=rolling_window).mean() /
                            strategy_returns.rolling(window=rolling_window).std()) * np.sqrt(252)

    # 绘制滚动夏普比率
    plt.figure(figsize=(10, 6))
    rolling_sharpe_ratio.plot()
    plt.title('Rolling 6-Month Sharpe Ratio')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.show()

    # 绘制最大回撤
    plt.figure(figsize=(10, 6))
    drawdown.plot()
    plt.title('Maximum Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.axhline(max_drawdown, color='red', linestyle='--', label='Max Drawdown')
    plt.legend()
    plt.grid(True)
    plt.show()