'''
Test klasy YFinanceDownloader.
'''

import pytest
from scripts.yahoofinanse_downloader import YFinanceDownloader
from scripts.essentials import calculate_technical_indicators

def test_preprocess_data():

    ticker_list = ['BA']
    begin_date = "20200101"
    end_date = "20200103"

    df = None

    try:
        yfd = YFinanceDownloader(ticker_list,begin_date,end_date)
        df = yfd.create_df()  
    except:
        print('Missing YFinance')

    assert df is not None

    df_result = calculate_technical_indicators(df, 'Date', 'BA_Open', 'BA_High', 'BA_Low', 'BA_Close', 'BA_Volume')
    print(df_result.shape)

    assert df_result.shape == (3,17)

