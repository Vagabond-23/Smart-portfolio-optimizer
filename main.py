import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from optimizer import PortfolioOptimizer

TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
RISK_FREE_RATE = 0.01
NUM_PORTFOLIOS = 10000

optimizer = PortfolioOptimizer(TICKERS, START_DATE, END_DATE, RISK_FREE_RATE)

results = optimizer.simulate_portfolios(NUM_PORTFOLIOS)
optimal = optimizer.get_optimal_portfolios(results)

metrics = optimizer.calculate_risk_metrics(optimal['max_sharpe']['weights'])

print("\nRisk Metrics (Max Sharpe Portfolio)")
print("Sortino Ratio:", metrics["sortino"])
print("Max Drawdown:", metrics["max_drawdown"])
print("Value at Risk (95%):", metrics["var_95"])

returns = np.array(results['returns'])
volatility = np.array(results['volatility'])
sharpe = np.array(results['sharpe'])

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    volatility,
    returns,
    c=sharpe,
    cmap='viridis',
    alpha=0.6
)

plt.colorbar(scatter, label="Sharpe Ratio")

plt.scatter(
    optimal['max_sharpe']['volatility'],
    optimal['max_sharpe']['return'],
    color='red',
    marker='*',
    s=300,
    label='Max Sharpe'
)

plt.scatter(
    optimal['min_volatility']['volatility'],
    optimal['min_volatility']['return'],
    color='blue',
    marker='*',
    s=300,
    label='Min Volatility'
)

# Capital Allocation Line
max_sharpe = optimal['max_sharpe']
cal_x = np.linspace(0, max_sharpe['volatility']*1.5, 100)
cal_y = RISK_FREE_RATE + max_sharpe['sharpe'] * cal_x

plt.plot(cal_x, cal_y, linestyle="--", color="black", label="Capital Allocation Line")

plt.title("Efficient Frontier - Smart Portfolio Optimizer")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")
plt.legend()
plt.grid(True)

plt.savefig("efficient_frontier.png", dpi=300)
plt.show()

print("\nOptimal Portfolio Weights")

print("\nMax Sharpe Portfolio")
for ticker, weight in zip(TICKERS, optimal['max_sharpe']['weights']):
    print(f"{ticker}: {weight:.2%}")

print("\nMin Volatility Portfolio")
for ticker, weight in zip(TICKERS, optimal['min_volatility']['weights']):
    print(f"{ticker}: {weight:.2%}")

# Portfolio Allocation Pie Chart
plt.figure(figsize=(6,6))

plt.pie(
    optimal['max_sharpe']['weights'],
    labels=TICKERS,
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Optimal Portfolio Allocation")

plt.savefig("allocation.png", dpi=300)
plt.show()

# Backtest
performance = optimizer.backtest_portfolio(
    optimal['max_sharpe']['weights']
)

# Benchmark
benchmark = yf.download("SPY", start=START_DATE, end=END_DATE)

if "Adj Close" in benchmark.columns:
    benchmark = benchmark["Adj Close"]
else:
    benchmark = benchmark["Close"]

benchmark_returns = benchmark.pct_change().dropna()
benchmark_cumulative = (1 + benchmark_returns).cumprod()

# Align dates
benchmark_cumulative = benchmark_cumulative.loc[performance.index]

plt.figure(figsize=(10,6))

plt.plot(performance, label="Optimized Portfolio")
plt.plot(benchmark_cumulative, label="S&P500 (SPY)", linestyle="--")

plt.title("Portfolio vs Market Benchmark")
plt.xlabel("Date")
plt.ylabel("Growth of $1 Investment")
plt.legend()
plt.grid(True)

plt.savefig("backtest_vs_sp500.png", dpi=300)
plt.show()

# Monte Carlo Forecast
simulations = optimizer.monte_carlo_forecast(
    optimal['max_sharpe']['weights']
)

mean_path = simulations.mean(axis=0)
lower = np.percentile(simulations, 5, axis=0)
upper = np.percentile(simulations, 95, axis=0)

plt.figure(figsize=(10,6))

for sim in simulations[:30]:
    plt.plot(sim, alpha=0.2)

plt.plot(mean_path, color='black', linewidth=2, label="Expected Path")
plt.fill_between(range(len(mean_path)), lower, upper, alpha=0.2)

plt.title("Monte Carlo Portfolio Forecast")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)

plt.savefig("monte_carlo_forecast.png", dpi=300)
plt.show()