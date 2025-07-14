import streamlit as st
import pandas as pd
import numpy as np
from keras import models
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from web_scrape import scrape_yahoo_finance
from model import add_technical_indicators  # Import feature engineering

st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID", "GOOG")

end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# --- Data Loading ---
stock_data = scrape_yahoo_finance(stock)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)
stock_data['Adj Close'] = pd.to_numeric(stock_data['Adj Close'], errors='coerce')

model = models.load_model("Latest_stock_price_model.keras")

st.subheader("Stock Data and Moving Averages")
col1, col2 = st.columns([1, 2])

ma_options = [100, 200, 250]
for ma in ma_options:
    col_name = f'MA_for_{ma}_days'
    stock_data[col_name] = stock_data["Adj Close"].rolling(ma).mean()

with col1:
    st.write(stock_data)

with col2:
    fig = plt.figure(figsize=(10, 6))
    plt.plot(stock_data["Adj Close"], label="Original")
    for ma in ma_options:
        col_name = f'MA_for_{ma}_days'
        plt.plot(stock_data[col_name], label=f"MA {ma} Days")
    plt.legend()
    plt.title('Stock Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)

# --- Feature Engineering and Prediction ---
feature_cols = [
    'Adj Close', 'Moving Avg', 'Percent Change',
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'RSI',
    'MACD', 'MACD_Signal',
    'Lag_1', 'Lag_2', 'Lag_3',
    'DayOfWeek', 'DayOfMonth'
]

stock_data = add_technical_indicators(stock_data)
stock_data.dropna(subset=feature_cols, inplace=True)

scaler_X = joblib.load('feature_scaler.save')
scaler_y = joblib.load('target_scaler.save')

X = stock_data[feature_cols].values
X_scaled = scaler_X.transform(X)

seq_len = 30
N_FUTURE = 7  # Number of days to forecast

# Rolling forecast
last_window = X_scaled[-seq_len:].copy()
reg_preds = []
predicted_prices = []
future_dates = []
last_date = stock_data.index[0]  # Changed from [-1] to [0] since data is sorted newest first
last_price = stock_data['Adj Close'].iloc[0]  # Changed from [-1] to [0]

for i in range(N_FUTURE):
    input_seq = last_window.reshape(1, seq_len, -1)
    reg_pred_scaled, _ = model.predict(input_seq)
    reg_pred = scaler_y.inverse_transform(reg_pred_scaled)[0, 0]
    reg_preds.append(reg_pred)
    
    # Calculate predicted price
    predicted_price = last_price * (1 + reg_pred)
    predicted_prices.append(predicted_price)
    
    # Generate next date (skip weekends)
    next_date = last_date + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_date += pd.Timedelta(days=1)
    future_dates.append(next_date)
    last_date = next_date
    last_price = predicted_price
    
    # Prepare next input window (simplified)
    next_row = last_window[-1].copy()
    next_row[feature_cols.index('Adj Close')] = scaler_X.transform([[predicted_price] + [0]*(len(feature_cols)-1)])[0][0]
    next_row[feature_cols.index('Percent Change')] = scaler_X.transform([[0, 0, reg_pred] + [0]*(len(feature_cols)-3)])[0][2]
    last_window = np.vstack([last_window[1:], next_row])

# Calculate predicted moving averages
ma_100_pred = []
ma_200_pred = []
ma_250_pred = []

# Get historical prices for MA calculation
historical_prices = stock_data['Adj Close'].tolist()

for i in range(N_FUTURE):
    # Add predicted price to historical list
    historical_prices.append(predicted_prices[i])
    
    # Calculate moving averages
    if len(historical_prices) >= 100:
        ma_100_pred.append(np.mean(historical_prices[-100:]))
    else:
        ma_100_pred.append(np.nan)
        
    if len(historical_prices) >= 200:
        ma_200_pred.append(np.mean(historical_prices[-200:]))
    else:
        ma_200_pred.append(np.nan)
        
    if len(historical_prices) >= 250:
        ma_250_pred.append(np.mean(historical_prices[-250:]))
    else:
        ma_250_pred.append(np.nan)

results_df = pd.DataFrame({
    'Predicted Price': predicted_prices,
    'Predicted % Change': [f"{pct:.4f}" for pct in reg_preds],
    'MA 100': [f"{ma:.2f}" if not np.isnan(ma) else "N/A" for ma in ma_100_pred],
    'MA 200': [f"{ma:.2f}" if not np.isnan(ma) else "N/A" for ma in ma_200_pred],
    'MA 250': [f"{ma:.2f}" if not np.isnan(ma) else "N/A" for ma in ma_250_pred]
}, index=future_dates)

st.subheader('7-Day Stock Forecast')
st.write("**Predicted stock prices, percent changes, and moving averages for the next 7 trading days:**")
st.write(results_df)

# Plot predictions
fig2 = plt.figure(figsize=(12, 6))
plt.plot(results_df.index, predicted_prices, 'b-o', label='Predicted Price', linewidth=2, markersize=6)
plt.plot(results_df.index, [float(ma) if ma != "N/A" else np.nan for ma in results_df['MA 100']], 'r--', label='MA 100', alpha=0.7)
plt.plot(results_df.index, [float(ma) if ma != "N/A" else np.nan for ma in results_df['MA 200']], 'g--', label='MA 200', alpha=0.7)
plt.plot(results_df.index, [float(ma) if ma != "N/A" else np.nan for ma in results_df['MA 250']], 'y--', label='MA 250', alpha=0.7)
plt.legend()
plt.title('7-Day Stock Price Forecast with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
st.pyplot(fig2)

# Investment insights
st.subheader('Investment Insights')
latest_price = stock_data['Adj Close'].iloc[0]  # Changed from [-1] to [0]
predicted_end_price = predicted_prices[-1]
total_return = (predicted_end_price - latest_price) / latest_price * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"${latest_price:.2f}")
with col2:
    st.metric("Predicted Price (7 days)", f"${predicted_end_price:.2f}")
with col3:
    st.metric("Expected Return", f"{total_return:.2f}%")

# Technical analysis summary
st.subheader('Technical Analysis Summary')
if total_return > 0:
    st.success(f"📈 **Bullish Signal**: Model predicts a {total_return:.2f}% increase over 7 days")
else:
    st.error(f"📉 **Bearish Signal**: Model predicts a {abs(total_return):.2f}% decrease over 7 days")

# Moving average analysis
if len(ma_100_pred) > 0 and not np.isnan(ma_100_pred[-1]):
    current_ma100 = stock_data['Moving Avg'].iloc[0]  # Changed from [-1] to [0]
    pred_ma100 = ma_100_pred[-1]
    if predicted_end_price > pred_ma100:
        st.info("✅ **MA Analysis**: Predicted price above 100-day moving average - potential support level")
    else:
        st.warning("⚠️ **MA Analysis**: Predicted price below 100-day moving average - potential resistance level")