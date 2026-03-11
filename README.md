# 📈 Smart Portfolio Optimizer

An interactive **portfolio optimization dashboard** built with **Python and Streamlit** that applies **Modern Portfolio Theory (MPT)** to identify optimal stock allocations based on risk and return.

The application simulates thousands of portfolios, constructs the **Efficient Frontier**, identifies optimal portfolios, and evaluates performance using risk metrics and Monte Carlo forecasting.

---

# 🚀 Features

* 📊 **Efficient Frontier Visualization**
* ⭐ **Maximum Sharpe Ratio Portfolio**
* 🔵 **Minimum Volatility Portfolio**
* 🥧 **Optimal Portfolio Allocation Pie Chart**
* 📉 **Portfolio Backtesting**
* 📊 **Benchmark Comparison with S&P 500**
* ⚠️ **Risk Metrics**

  * Sortino Ratio
  * Maximum Drawdown
  * Value at Risk (95%)
* 🔮 **Monte Carlo Portfolio Forecast**
* 🖥️ **Interactive Dashboard using Streamlit**

---

# 🧠 Concepts Used

This project implements key ideas from **quantitative finance and portfolio theory**.

### Expected Return

The expected return of an asset is the mean of historical returns.

[
\mu = \frac{1}{N}\sum_{i=1}^{N} r_i
]

---

### Portfolio Volatility

[
\sigma_p = \sqrt{w^T \Sigma w}
]

Where:

* (w) = portfolio weights
* (\Sigma) = covariance matrix of asset returns

---

### Sharpe Ratio

Measures risk-adjusted return.

[
S = \frac{R_p - R_f}{\sigma_p}
]

Where:

* (R_p) = portfolio return
* (R_f) = risk-free rate
* (\sigma_p) = portfolio volatility

---

### Efficient Frontier

The **Efficient Frontier** represents portfolios that maximize expected return for a given level of risk.

Portfolios below the frontier are **suboptimal**.

---

# 🛠 Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* yfinance
* Streamlit

---

# 📂 Project Structure

```
Smart-Portfolio-Optimizer
│
├── app.py              # Streamlit dashboard
├── optimizer.py        # Portfolio optimization logic
├── main.py             # Script version (optional)
├── requirements.txt
└── README.md
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Smart-Portfolio-Optimizer.git
cd Smart-Portfolio-Optimizer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

The app will automatically open in your browser.

---

# ⚙️ How It Works

1. Fetch historical stock data using **Yahoo Finance**
2. Calculate daily returns and covariance matrix
3. Generate **10,000+ random portfolios**
4. Compute:

   * expected return
   * volatility
   * Sharpe ratio
5. Identify:

   * Maximum Sharpe portfolio
   * Minimum volatility portfolio
6. Visualize results through the **Efficient Frontier**
7. Backtest portfolio performance vs **S&P 500**
8. Forecast future portfolio performance using **Monte Carlo simulation**

---

# 📸 Example Output

The dashboard includes:

* Efficient frontier visualization
* Optimal portfolio allocation
* Portfolio backtesting vs benchmark
* Risk metrics
* Monte Carlo forecast

*(Add screenshots here after running the app)*

---

# ⚠️ Notes

* Stock tickers must use **official ticker symbols** (e.g., `TSLA`, not `TESLA`).
* Data is fetched from **Yahoo Finance via yfinance**.
* Portfolio simulations use **Monte Carlo sampling**.

---

# 📚 Future Improvements

Potential extensions:

* Portfolio constraints (max allocation per asset)
* Sector diversification constraints
* Real-time market data
* Portfolio rebalancing strategies
* Deployment as a cloud web application

---

# 👨‍💻 Author

Developed by **Rakshit Dahiya**

---
