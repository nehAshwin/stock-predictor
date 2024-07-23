import streamlit as st 
from web_scrape import *
import pandas as pd
import numpy as np
from keras import models
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID", "GOOG")

end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# convert end and start to correct code to input into web_scrape function

stock_data = scrape_yahoo_finance(stock)

stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)
stock_data['Adj Close'] = pd.to_numeric(stock_data['Adj Close'], errors='coerce')

model = models.load_model("Latest_stock_price_model.keras")

st.subheader("Stock Data")
st.write(stock_data)

splitting_len = int(len(stock_data) * 0.7)
x_test = pd.DataFrame(stock_data['Adj Close'][splitting_len:])

def plot_graph(figsize, values, full_data):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data['Adj Close'], 'b')
    return fig

st.subheader("Original Adj Close Price and MA for 250 Days")
stock_data['MA_for_250_days'] = stock_data["Adj Close"].rolling(250).mean()
st.pyplot(plot_graph((15,6), stock_data["MA_for_250_days"], stock_data))

st.subheader("Original Adj Close Price and MA for 200 Days")
stock_data['MA_for_200_days'] = stock_data["Adj Close"].rolling(200).mean()
st.pyplot(plot_graph((15,6), stock_data["MA_for_200_days"], stock_data))

st.subheader("Original Adj Close Price and MA for 100 Days")
stock_data['MA_for_100_days'] = stock_data["Adj Close"].rolling(100).mean()
st.pyplot(plot_graph((15,6), stock_data["MA_for_100_days"], stock_data))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Adj Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1), 'predictions': inv_pre.reshape(-1)
    }, index = stock_data.index[splitting_len +100:]
) 
st.subheader('Original Values vs Predicted Values')
st.write(plotting_data)

st.subheader("Original Adj Close Price vs Predicted Adj Close Price")
fig = plt.figure(figsize=(15,6))

plt.plot(pd.concat([stock_data['Adj Close'][:splitting_len+100], plotting_data], axis=0))

plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])

st.pyplot(fig)