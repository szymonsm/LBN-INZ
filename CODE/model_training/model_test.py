import pytest
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from datetime import datetime
import yfinance as yf
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

def test_predict_data():
    model_path = 'models/test_model.joblib'

    loaded_model = None
    try:
        loaded_model = joblib.load(model_path)
    except:
        print('Model not loaded')
    
    assert loaded_model is not None
    
    input_data = pd.DataFrame({
        'ema_20': [1.0],
        'rsi_14': [1.0],
        'slowk': [1.0],
        'vwap': [1.0],
        'roc_14': [1.0]
    })

    predictions = loaded_model.predict(input_data)

    assert isinstance(predictions, np.ndarray)

    assert np.round(predictions[0],2) == -0.11

