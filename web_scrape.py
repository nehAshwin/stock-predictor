import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def scrape_yahoo_finance(ticker):

    # Set start and end times dynamically
    start = 1090540800  # keep as earliest possible
    end = int(datetime.now().timestamp())

    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={end}&interval=1d'
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    
    #HTTP GET request to finance.yahoo
    #headers is necessary to make client recognizable by server
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:

        data = response.json()      # Yahoo finance API call automatically sends a JSON file
        
        # CHANGED: eliminated parsing w BeautifulSoup, instead parse JSON
        try:

            # pull specific columns
            timestamps = data['chart']['result'][0]['timestamp']
            adjcloses = data['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']

            # convert to panda dataframe
            df = pd.DataFrame({'Date': pd.to_datetime(timestamps, unit='s'),
                               'Adj Close': adjcloses})
            
            # drop all missing values in data fram (apply changes to og df)
            # QUESTION: is there a better way to handle missing information?
            df.dropna(inplace=True)
            df.sort_values('Date', ascending=False, inplace=True)  # Most recent first
            return df
        
        except (KeyError, IndexError, TypeError):
            print("Unexpected JSON structure")
        
    else:
        print(f"Failed to retrieve data for {ticker}")
        print(response.status_code)
        return pd.DataFrame(columns=['Date', 'Adj Close'])
    
def plot_graph(figsize, values, column_name):
    plt.figure(figsize = figsize)
    values.plot()
    plt.xlabel("years")
    plt.ylabel(column_name)
    # CHANGE TO ticker
    plt.title(f"{column_name} of Google data")
    plt.show() 

# print(scrape_yahoo_finance('AAPL'))