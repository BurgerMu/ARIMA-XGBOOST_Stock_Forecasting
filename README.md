# ARIMA-XGBOOST_Stock_Forecasting
# 📈 S&P 500 Time-Series & Machine Learning Forecasting Engine (1927–2026)

An end-to-end quantitative time-series forecasting and machine learning pipeline for predicting monthly S&P 500 (`^GSPC`) index closing levels over a 24-month horizon. This project contrasts classical econometric modeling (**auto-ARIMA / SARIMAX**) with modern ensemble gradient boosting (**XGBoost**) using nearly a century of historical data (1928 to present).

---

## 📌 Executive Summary

* **Objective:** Benchmark classical statistical models against machine learning regressors for long-horizon (24-month) financial time-series forecasting.
* **Dataset:** Monthly closing levels of the S&P 500 Index from **February 1928 to October 2026**.
* **Train / Test Split:**
  * **Train Set:** Feb 1928 – Jun 2023 (~733 months)
  * **Test Set:** Aug 2023 – Oct 2026 (24-month out-of-sample forecast window)
* **Key Findings:** 
  * Long-term log-linear regression reveals a market **CAGR of 11.13%** with an $R^2 = 0.9655$.
  * XGBoost captures short-term autoregressive dynamics effectively but requires explicit trend decomposition for multi-step extrapolation.
  * ARIMA automatically handles non-stationarity via optimal differencing ($d$).

---

## 🏗️ Architecture & Pipeline Workflows

```
  ┌────────────────────────────────────────────────────────┐
  │         Raw S&P 500 Monthly Data (1928 - 2026)         │
  └───────────────────────────┬────────────────────────────┘
                              │
               [Cleaning & Resampling ('MS')]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
    [Train Set: 1928–2023]        [Test Set: 2023–2026]
               │                             │
       ┌───────┴────────┐                    │
       ▼                ▼                    │
  [auto-ARIMA]    [XGBoost Engine]           │
       │                │                    │
       │       (Feature Engineering:         │
       │        Lags, Rolling Vol,           │
       │        Recursive Predict)           │
       └───────┬────────┘                    │
               ▼                             ▼
     [24-Month Predictions] ────> [Out-of-Sample Evaluation]
                                   (MAE, RMSE, MAPE, R²)
```

---

## 🛠️ Key Technical Features

### 1. Quantitative Exploratory Data Analysis (EDA)
* **Log-Linear Growth Fit:** Fitted exponential trend parameters ($\ln(Y_t) = \alpha + \beta t$) establishing historical CAGR baseline.
* **Historical Drawdown Analysis:** Calculated peak-to-trough stress metrics, including the maximum historical drawdown (-85.3% during the Great Depression).
* **Volatilty Analysis:** Measured historical annualized rolling volatility ($\sigma_{\text{ann}} \approx 16.2\%$).

### 2. Time-Series Econometrics (ARIMA / SARIMAX)
* Stationarity analysis via **Augmented Dickey-Fuller (ADF)** testing.
* Automated hyperparameter tuning using **`pmdarima.auto_arima`** to minimize Akaike Information Criterion (AIC) / Bayesian Information Criterion (BIC).

### 3. Machine Learning Pipeline (XGBoost Regressor)
* **Feature Engineering:**
  * **Autoregressive Lags:** $Y_{t-1}, Y_{t-2}, \dots, Y_{t-k}$
  * **Rolling Window Statistics:** Moving averages and rolling standard deviations over multiple windows (3, 6, 12 months) to capture regime shifts and dynamic volatility clustering.
  * **Temporal Features:** Year, Month, and Quarter signals.
* **Recursive Multi-Step Forecasting:** Iterative prediction loop where $t+1$ forecasts feed back into feature generation for $t+2$ steps up to 24 months.

---

## 📊 Evaluation Metrics

Models are evaluated out-of-sample over the 24-month test window using standard quantitative performance metrics:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 💻 Tech Stack

* **Language:** Python 3.9+
* **Data Manipulation:** `pandas`, `numpy`
* **Modeling & ML:** `statsmodels`, `pmdarima`, `xgboost`, `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`, `plotly`

---

## 🚀 Quickstart & Usage

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/sp500-forecasting-engine.git
cd sp500-forecasting-engine
pip install -r requirements.txt
```

### 2. Run the Notebook or Script
```bash
jupyter notebook sp500_forecasting.ipynb
```
*or execute directly via Python:*
```bash
python main.py
```

---

## 📂 Project Structure

```text
├── data/
│   └── sp500_monthly.csv       # S&P 500 price history (1928-2026)
├── notebooks/
│   └── sp500_forecasting.ipynb # Complete analysis & execution notebook
├── src/
│   ├── data_loader.py          # Resampling and data preprocessing
│   ├── feature_engineering.py  # Lag creation and rolling stats generator
│   └── models.py               # ARIMA and XGBoost training/predict loops
├── requirements.txt            # Dependencies list
├── README.md                   # Project documentation
└── LICENSE                     # MIT License
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
