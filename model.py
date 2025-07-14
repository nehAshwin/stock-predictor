import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from web_scrape import scrape_yahoo_finance, plot_graph

# --- Feature Engineering ---
def add_technical_indicators(df):
    window = 20
    df['Moving Avg'] = df['Adj Close'].rolling(window).mean()
    df['BB_Middle'] = df['Adj Close'].rolling(window).mean()
    df['BB_Std'] = df['Adj Close'].rolling(window).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    # RSI
    window_rsi = 14
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window_rsi).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Percent Change
    df['Percent Change'] = df['Adj Close'].pct_change()
    # Lagged percent changes
    for lag in range(1, 4):
        df[f'Lag_{lag}'] = df['Percent Change'].shift(lag)
    # Day of week/month
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    return df

# --- Main Training Pipeline ---

def main():
    # Load data
    df = scrape_yahoo_finance('GOOG')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')

    # Add features
    df = add_technical_indicators(df)
    feature_cols = [
        'Adj Close', 'Moving Avg', 'Percent Change',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'RSI',
        'MACD', 'MACD_Signal',
        'Lag_1', 'Lag_2', 'Lag_3',
        'DayOfWeek', 'DayOfMonth'
    ]
    df.dropna(subset=feature_cols, inplace=True)

    # Target variable
    df['Next_Percent_Change'] = df['Percent Change'].shift(-1)
    df.dropna(subset=['Next_Percent_Change'], inplace=True)

    # Prepare features and target
    X = df[feature_cols].values
    y = df['Next_Percent_Change'].values

    # Scale features
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

    # Build Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f'Train R² Score: {train_score:.4f}')
    print(f'Test R² Score: {test_score:.4f}')

    # Predict
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2))
    print(f'RMSE: {rmse:.6f}')

    # Save model and scalers
    joblib.dump(model, 'stock_prediction_model.pkl')
    joblib.dump(scaler_X, 'feature_scaler.save')
    joblib.dump(scaler_y, 'target_scaler.save')
    
    print("Model and scalers saved successfully!")

if __name__ == '__main__':
    main()