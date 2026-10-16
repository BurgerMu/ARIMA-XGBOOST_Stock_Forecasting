# ARIMA-XGBOOST_Stock_Forecasting
# S&P 500 Stock Market Forecasting: Traditional Statistics vs. Machine Learning

[![Open In Colab](https://colab.research.google.com/drive/1CagD6sNPB9wwc7uGLVzKv8xuKUh5ak-a?usp=sharing)

---

## Project Overview
I wanted to see if we can accurately forecast where the stock market—specifically the S&P 500 index—is heading over a 2-year (24-month) horizon. 

Using nearly 100 years of historical monthly market data (from 1928 to 2026), I built and compared two different prediction approaches:
1. ARIMA: A classical statistical model designed to find patterns in time-series data.
2. XGBoost: A modern machine learning algorithm that uses decision trees to learn complex relationships.

---

## Why Stock Market Prediction Is Challenging
Predicting the stock market is difficult because stock prices generally trend upward over long periods, but they also experience sudden crashes and market shocks (like the 1929 Great Depression or recent market volatility).

* The Challenge: Machine learning models like XGBoost cannot easily predict numbers higher than what they saw in training unless we engineer special features for them.
* The Goal: Clean and transform the data so both models can make realistic predictions without getting thrown off by long-term growth trends.

---

## Project Workflow (Step-by-Step)

1. Data Cleaning and Alignment:
   I pulled historical monthly stock prices and cleaned up missing days (such as weekends and holidays) so every row represents the first day of the month.

2. Feature Engineering:
   To give the machine learning model context about past trends, I created helper features:
   * Lag Features: Giving the model prices from 1, 2, or 3 months prior.
   * Rolling Averages: Calculating average prices and standard deviations over recent months to capture market momentum and volatility.

3. Train/Test Split:
   * Training Period: 1928 to June 2023 (Used to teach the models).
   * Testing Period: August 2023 to October 2026 (24 months reserved for evaluating out-of-sample predictions).

4. Model Evaluation:
   I compared predicted values against actual stock prices using standard metrics including MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and MAPE (Mean Absolute Percentage Error).

---

## Main Key Takeaways

* Compound Growth: Over the past century, the S&P 500 has maintained an average growth rate of approximately 11.13% per year despite historical downturns.
* Stats vs. Machine Learning: ARIMA is strong at tracking overall long-term trends, while XGBoost excels at learning local momentum when provided with engineered lag features.
* Multi-Step Drift: Predicting 1 month into the future is straightforward, but projecting 24 months ahead causes small forecasting errors to compound over time.

---

## How to Run This Project

### Option 1: Open in Google Colab (Recommended)
1. Click the "Open In Colab" badge at the top of this page.
2. Sign in with your Google account.
3. Click "Runtime" in the top menu and select "Run all".

### Option 2: Run Locally
1. Clone this repository to your computer:
   git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
2. Install the required Python packages:
   pip install pandas numpy matplotlib statsmodels pmdarima xgboost scikit-learn
3. Open and run the Jupyter Notebook in your environment.
