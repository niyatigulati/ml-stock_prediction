import requests
import os
import io
import pandas as pd
import numpy as np
alpha_vantage_url ='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey=1YB64755JWVBCS3&datatype=csv'
api_key = '1YB64755JWVBCS3'

def get_alpha_vantage_data(ABC):
    
    
    url = alpha_vantage_url.format(ABC)
    print (url)
    content = requests.get(url).content
    data = pd.read_csv(io.StringIO(content.decode('utf-8')))
    print(data.head())
    print(data.shape)
    print("\n")
    print("Open   --- mean :", np.mean(data['open']),  "  \t Std: ", np.std(data['open']),  "  \t Max: ", np.max(data['open']),  "  \t Min: ", np.min(data['open']))
    print("High   --- mean :", np.mean(data['high']),  "  \t Std: ", np.std(data['high']),  "  \t Max: ", np.max(data['high']),  "  \t Min: ", np.min(data['high']))
    print("Low    --- mean :", np.mean(data['low']),   "  \t Std: ", np.std(data['low']),   "  \t Max: ", np.max(data['low']),   "  \t Min: ", np.min(data['low']))
    print("Close  --- mean :", np.mean(data['close']), "  \t Std: ", np.std(data['close']), "  \t Max: ", np.max(data['close']), "  \t Min: ", np.min(data['close']))
    print("Volume --- mean :", np.mean(data['volume']),"  \t Std: ", np.std(data['volume']),"  \t Max: ", np.max(data['volume']),"  \t Min: ", np.min(data['volume']))

    import preprocess_data as ppd
    stocks = ppd.remove_data(data)

    #Print the dataframe head and tail
    print(stocks.head())
    print("---")
    print(stocks.tail())

    # import visualize

    # visualize.plot_basic(stocks)

    stocks = ppd.get_normalised_data(stocks)
    print(stocks.head())

    print("\n")
    print("open   --- mean :", np.mean(stocks['open']),  "  \t Std: ", np.std(stocks['open']),  "  \t Max: ", np.max(stocks['open']),  "  \t Min: ", np.min(stocks['open']))
    print("close  --- mean :", np.mean(stocks['close']), "  \t Std: ", np.std(stocks['close']), "  \t Max: ", np.max(stocks['close']), "  \t Min: ", np.min(stocks['close']))
    print("volume --- mean :", np.mean(stocks['volume']),"  \t Std: ", np.std(stocks['volume']),"  \t Max: ", np.max(stocks['volume']),"  \t Min: ", np.min(stocks['volume']))
    # visualize.plot_basic(stocks)
    # stocks.to_csv('preprocessed.csv',index= False)
    # check poin 2
    return stocks