import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Input, Model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
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

    # Multi-output labels
    df['Next_Percent_Change'] = df['Percent Change'].shift(-1)
    df['Next_Direction'] = (df['Next_Percent_Change'] > 0).astype(int)
    df.dropna(subset=['Next_Percent_Change', 'Next_Direction'], inplace=True)

    # Prepare features and labels
    X = df[feature_cols].values
    y_reg = df['Next_Percent_Change'].values.reshape(-1, 1)
    y_clf = df['Next_Direction'].values.reshape(-1, 1)

    # Scale features
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_reg_scaled = scaler_y.fit_transform(y_reg)

    # Sequence data for LSTM
    seq_len = 30
    X_seq, y_reg_seq, y_clf_seq = [], [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_len:i])
        y_reg_seq.append(y_reg_scaled[i])
        y_clf_seq.append(y_clf[i])
    X_seq, y_reg_seq, y_clf_seq = np.array(X_seq), np.array(y_reg_seq), np.array(y_clf_seq)

    # Train/test split
    split = int(len(X_seq) * 0.7)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_reg_train, y_reg_test = y_reg_seq[:split], y_reg_seq[split:]
    y_clf_train, y_clf_test = y_clf_seq[:split], y_clf_seq[split:]

    # Build multi-output model
    input_layer = Input(shape=(seq_len, X_seq.shape[2]))
    x = LSTM(64, return_sequences=True)(input_layer)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    # Regression head
    reg_head = Dense(16, activation='relu')(x)
    reg_out = Dense(1, name='reg_out')(reg_head)
    # Classification head
    clf_head = Dense(16, activation='relu')(x)
    clf_out = Dense(1, activation='sigmoid', name='clf_out')(clf_head)
    model = Model(inputs=input_layer, outputs=[reg_out, clf_out])
    model.compile(
        optimizer='adam',
        loss={'reg_out': 'mse', 'clf_out': 'binary_crossentropy'},
        metrics={'reg_out': 'mae', 'clf_out': 'accuracy'}
    )
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, {'reg_out': y_reg_train, 'clf_out': y_clf_train},
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=2
    )
    # Evaluate
    results = model.evaluate(X_test, {'reg_out': y_reg_test, 'clf_out': y_clf_test}, verbose=2)
    print('Test loss/MAE/Accuracy:', results)
    # Predict
    reg_pred_scaled, clf_pred = model.predict(X_test)
    reg_pred = scaler_y.inverse_transform(reg_pred_scaled)
    # Save model and scalers
    model.save('Latest_stock_price_model.keras')
    joblib.dump(scaler_X, 'feature_scaler.save')
    joblib.dump(scaler_y, 'target_scaler.save')
    # Optionally plot
    # plotting_data = pd.DataFrame({'reg_pred': reg_pred.reshape(-1), 'clf_pred': clf_pred.reshape(-1)}, index=df.index[split+seq_len:])
    # plot_graph((15,6), plotting_data, 'Predictions')

if __name__ == '__main__':
    main()