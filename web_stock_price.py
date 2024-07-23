import streamlit as st 
from web_scrape import *
import pandas as pd
import numpy as np
from keras import models
import matplotlib.pyplot as plt
from datetime import datetime

st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID", "GOOG")

end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# convert end and start to correct code to input into web_scrape function

stock_data = scrape_yahoo_finance(stock)

model = models.load_model("Latest_stock_price_model.keras")

st.subheader("Stock Data")
st.write(stock_data)

splitting_len = int(len(stock_data) * 0.7)
x_test = pd.DataFrame(stock_data['Adj Close'][splitting_len:])

st.subheader("Original Adj Close Price and MA for 250 Days")
stock_data['MA_for_250_days'] = stock_data["Adj Close"].rolling(250).mean()
st.pyplot(plot_graph((15,6), stock_data["MA_for_250_days"], stock_data ))

st.subheader("Original Adj Close Price and MA for 200 Days")
stock_data['MA_for_200_days'] = stock_data["Adj Close"].rolling(200).mean()
st.pyplot(plot_graph((15,6), stock_data["MA_for_200_days"], stock_data ))

st.subheader("Original Adj Close Price and MA for 100 Days")
stock_data['MA_for_100_days'] = stock_data["Adj Close"].rolling(100).mean()
st.pyplot(plot_graph((15,6), stock_data["MA_for_100_days"], stock_data ))