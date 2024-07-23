from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt

def scrape_yahoo_finance(ticker):
    # goal for later: have the user input the current date and pull 5 years from there instead of fixed 5 years (check gpt)
    url = f'https://finance.yahoo.com/quote/{ticker}/history/?period1=1563909348&period2=1721762051'
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    #HTTP GET request to finance.yahoo
    #headers is necessary to make client recognizable by server
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')

        if table:
            rows = table.find_all('tr')[1:]  # Skip the header row
            data = []

            for row in rows:
                cols = row.find_all('td')

                if len(cols) < 6:
                    continue

                date = cols[0].text
                adj_close_price = cols[5].text.replace(',', '')
                data.insert(0, [date, adj_close_price])

            df = pd.DataFrame(data, columns=['Date', 'Adj Close'])
            # df['MA'] = df['Close'].rolling(5).mean()

            return df
        
        else:
            print(f"No data found for {ticker}")
            return pd.DataFrame(columns=['Date', 'Adj Close'])
        
    else:
        print(f"Failed to retrieve data for {ticker}")
        return pd.DataFrame(columns=['Date', 'Adj Close'])
    
def plot_graph(figsize, values, column_name):
    plt.figure(figsize = figsize)
    values.plot()
    plt.xlabel("years")
    plt.ylabel(column_name)
    plt.title(f"{column_name} of Google data")
    plt.show() 

print(scrape_yahoo_finance('AAPL'))
# plot_graph((15,5), scrape_yahoo_finance('GOOG')['Close'], 'Closing Price')
