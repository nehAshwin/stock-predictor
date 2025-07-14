# Stock Price Predictor

A machine learning-powered web application that predicts stock prices and provides investment insights using advanced technical indicators and LSTM neural networks.

## Features

### **7-Day Rolling Forecast**
- Predicts stock prices for the next 7 trading days
- Shows predicted percent changes and moving averages

### **Investment Insights**
- **Bullish/Bearish Signals**: Clear buy/sell recommendations
- **Expected Returns**: Predicted percentage gains/losses
- **Support/Resistance Analysis**: Moving average-based levels
- **Risk Assessment**: Technical analysis summary

## Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow/Keras (LSTM Neural Networks)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Data Source**: Yahoo Finance API
- **Visualization**: Matplotlib

## Access the App
Open your browser and go to `https://stock-predictor-7onn7zaabassngvxwcsggy.streamlit.app`

## How to Use

1. **Enter Stock Symbol**: Type any valid stock symbol (e.g., GOOG, AAPL, MSFT)
2. **View Historical Data**: See current stock data with moving averages
3. **Get Predictions**: View 7-day forecast with price predictions and technical analysis
4. **Make Decisions**: Use the investment insights to inform your trading decisions

## Project Structure

```
stock-predictor/
├── web_stock_price.py      # Main Streamlit application
├── model.py               # LSTM model training script
├── web_scrape.py          # Yahoo Finance data scraper
├── requirements.txt       # Python dependencies
├── Latest_stock_price_model.keras  # Trained model
├── feature_scaler.save    # Feature scaling parameters
├── target_scaler.save     # Target scaling parameters
└── README.md             # This file
```

## Disclaimer

**This application is for educational and research purposes only.**
- Stock predictions are not guaranteed and should not be the sole basis for investment decisions
- Always conduct thorough research and consider consulting with financial advisors
- Past performance does not indicate future results
- The model's accuracy may vary based on market conditions and data quality

