from web_scrape import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential 
from keras import layers

# load data
df = scrape_yahoo_finance('GOOG')

# Preprocess data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')

# Simple Moving Average per 20 days
df['Moving Avg'] = df['Adj Close'].rolling(20).mean()
df.dropna(inplace=True)

# Percent Change
df['Percent Change'] = df['Adj Close'].pct_change()

# print(df[['Adj Close', 'Percent Change']].head())

# Preprocessing Data
adj_close_price = df.loc[:, ['Adj Close']]
# print(max(adj_close_price.values), min(adj_close_price.values))
# print(type(adj_close_price))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(adj_close_price)
# print(scaled_data)

# Split Data

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

splitting_len = int(len(x_data) * 0.7)
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# Building Model (Neural Network)

model = Sequential()
model.add(layers.LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(64, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=2)

# model.summary()

# Predicting Test Data

predictions = model.predict(x_test)

inv_predictions = scaler.inverse_transform(predictions)
# print(inv_predictions)
inv_y_test = scaler.inverse_transform(y_test)
# print(inv_y_test)

rmse = np.sqrt(np.mean((inv_predictions - inv_y_test)**2))
# print(rmse)

plotting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1), 'predictions': inv_predictions.reshape(-1)
    }, index = df.index[splitting_len +100:]
)

# print(plotting_data.head)

plot_graph((15,6), pd.concat([adj_close_price[:splitting_len+100], plotting_data], axis=0), 'whole data')

model.save("Latest_stock_price_model.keras")