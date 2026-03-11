import pandas as pd
import numpy as np
import yfinance as yf


class PortfolioOptimizer:

    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.01):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate

        self.data = self._fetch_data()
        if self.data.empty:
            raise ValueError("No market data downloaded.")

        self.returns = self.data.pct_change().dropna()
        self.expected_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

    def _fetch_data(self):

        df = yf.download(self.tickers, start=self.start_date, end=self.end_date)

        if df.empty:
            return pd.DataFrame()

        if 'Adj Close' in df.columns.get_level_values(0):
            price = df['Adj Close']
        else:
            price = df['Close']

        return price.dropna()

    def simulate_portfolios(self, num_portfolios=10000):

        results = {
            "returns": [],
            "volatility": [],
            "sharpe": [],
            "weights": []
        }

        for _ in range(num_portfolios):

            weights = np.random.dirichlet(np.ones(len(self.tickers)))

            port_return = np.dot(weights, self.expected_returns)

            port_volatility = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix, weights))
            )

            sharpe = (port_return - self.risk_free_rate) / port_volatility

            results["returns"].append(port_return)
            results["volatility"].append(port_volatility)
            results["sharpe"].append(sharpe)
            results["weights"].append(weights)

        return results

    def get_optimal_portfolios(self, results):

        max_sharpe_idx = np.argmax(results["sharpe"])
        min_vol_idx = np.argmin(results["volatility"])

        return {
            "max_sharpe": {
                "return": results["returns"][max_sharpe_idx],
                "volatility": results["volatility"][max_sharpe_idx],
                "sharpe": results["sharpe"][max_sharpe_idx],
                "weights": results["weights"][max_sharpe_idx]
            },
            "min_volatility": {
                "return": results["returns"][min_vol_idx],
                "volatility": results["volatility"][min_vol_idx],
                "sharpe": results["sharpe"][min_vol_idx],
                "weights": results["weights"][min_vol_idx]
            }
        }

    def calculate_risk_metrics(self, weights):

        portfolio_returns = self.returns.dot(weights)

        if portfolio_returns.empty:
            return {"sortino": 0, "max_drawdown": 0, "var_95": 0}

        mean_return = portfolio_returns.mean()

        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0

        sortino = mean_return / downside_std if downside_std != 0 else 0

        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        var_95 = np.percentile(portfolio_returns, 5)

        return {
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "var_95": var_95
        }

    def backtest_portfolio(self, weights):

        portfolio_returns = self.returns.dot(weights)

        cumulative = (1 + portfolio_returns).cumprod()

        return cumulative

    def monte_carlo_forecast(self, weights, days=252, simulations=100):

        portfolio_returns = self.returns.dot(weights)

        mean = portfolio_returns.mean()
        std = portfolio_returns.std()

        simulations_data = []

        for _ in range(simulations):

            simulated_returns = np.random.normal(mean, std, days)

            path = (1 + simulated_returns).cumprod()

            simulations_data.append(path)

        return np.array(simulations_data)