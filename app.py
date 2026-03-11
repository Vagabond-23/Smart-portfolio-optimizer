import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import datetime
import yfinance as yf
from optimizer import PortfolioOptimizer

st.set_page_config(page_title="Smart Portfolio Optimizer", layout="wide")

st.title("📈 Smart Portfolio Optimizer")

st.sidebar.header("Configuration")

tickers = st.sidebar.text_input(
    "Stock Tickers (comma separated)",
    "AAPL,MSFT,GOOG,AMZN,META"
)

tickers = [t.strip().upper() for t in tickers.split(",")]

start_date = st.sidebar.date_input(
    "Start Date",
    datetime.date(2020,1,1)
)

end_date = st.sidebar.date_input(
    "End Date",
    datetime.date(2025,1,1)
)

num_portfolios = st.sidebar.slider(
    "Number of Portfolios",
    1000,
    20000,
    10000
)

risk_free_rate = st.sidebar.number_input(
    "Risk Free Rate",
    value=0.01
)

if st.sidebar.button("Run Optimization"):

    try:

        optimizer = PortfolioOptimizer(
            tickers,
            str(start_date),
            str(end_date),
            risk_free_rate
        )

    except Exception as e:
        st.error("Invalid ticker symbol. Please check your stock tickers.")
        st.stop()

    results = optimizer.simulate_portfolios(num_portfolios)

    optimal = optimizer.get_optimal_portfolios(results)

    returns = np.array(results["returns"])
    volatility = np.array(results["volatility"])
    sharpe = np.array(results["sharpe"])

    st.subheader("Efficient Frontier")

    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        volatility,
        returns,
        c=sharpe,
        cmap="viridis",
        alpha=0.6
    )

    ax.scatter(
        optimal["max_sharpe"]["volatility"],
        optimal["max_sharpe"]["return"],
        color="red",
        marker="*",
        s=300,
        label="Max Sharpe"
    )

    ax.scatter(
        optimal["min_volatility"]["volatility"],
        optimal["min_volatility"]["return"],
        color="blue",
        marker="*",
        s=300,
        label="Min Volatility"
    )

    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Portfolio Allocation")

    fig2, ax2 = plt.subplots()

    ax2.pie(
        optimal["max_sharpe"]["weights"],
        labels=tickers,
        autopct="%1.1f%%"
    )

    st.pyplot(fig2)

    st.subheader("Portfolio Backtest")

    performance = optimizer.backtest_portfolio(
        optimal["max_sharpe"]["weights"]
    )

    benchmark = yf.download(
        "SPY",
        start=str(start_date),
        end=str(end_date)
    )["Close"]

    benchmark_returns = benchmark.pct_change().dropna()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()

    fig3, ax3 = plt.subplots(figsize=(10,6))

    ax3.plot(performance, label="Optimized Portfolio")
    ax3.plot(benchmark_cumulative, label="S&P500", linestyle="--")

    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig3)


    st.subheader("Monte Carlo Portfolio Forecast")

    simulations = optimizer.monte_carlo_forecast(
        optimal["max_sharpe"]["weights"]
    )

    mean_path = simulations.mean(axis=0)
    lower = np.percentile(simulations, 5, axis=0)
    upper = np.percentile(simulations, 95, axis=0)

    fig4, ax4 = plt.subplots(figsize=(10,6))

    for sim in simulations[:30]:
        ax4.plot(sim, alpha=0.2)

    ax4.plot(mean_path, color="black", linewidth=2, label="Expected Path")
    ax4.fill_between(range(len(mean_path)), lower, upper, alpha=0.2)

    ax4.set_title("Monte Carlo Portfolio Forecast")
    ax4.set_xlabel("Days")
    ax4.set_ylabel("Portfolio Value")
    ax4.legend()
    ax4.grid(True)

    st.pyplot(fig4)


    st.subheader("Risk Metrics")

    metrics = optimizer.calculate_risk_metrics(
        optimal["max_sharpe"]["weights"]
    )

    stats = {
        "Expected Return": optimal["max_sharpe"]["return"],
        "Volatility": optimal["max_sharpe"]["volatility"],
        "Sharpe Ratio": optimal["max_sharpe"]["sharpe"],
        "Sortino Ratio": metrics["sortino"],
        "Max Drawdown": metrics["max_drawdown"],
        "VaR (95%)": metrics["var_95"]
    }

    st.table(stats)

