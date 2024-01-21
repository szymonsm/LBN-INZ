import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volume import VolumeWeightedAveragePrice

from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

def calculate_lagged_correlation(col1, col2, df, lag=14):
    """
    Calculate time-series correlation between two columns across a specified time lag.

    Parameters:
    - col1 (str): Name of the first column.
    - col2 (str): Name of the second column.
    - df (pd.DataFrame): The input DataFrame.
    - lag (int): Maximum lag for which to calculate correlation.

    Returns:
    - pd.Series: Series containing correlations for each lag.
    """
    # Create an empty DataFrame to store correlations
    correlations = pd.DataFrame(columns=['Lag', 'Correlation'])

    # Iterate over lags and calculate correlation for each lag shift
    for i in range(1, lag + 1):
        shifted_col2 = df[col2].shift(i)
        correlation = df[col1].corr(shifted_col2)

        # Save lag and correlation in the DataFrame

        correlations = pd.concat([correlations, pd.DataFrame({'Lag': [i], 'Correlation': [correlation]})], axis=0)

         

    return correlations

def replace_with_0(df,columns_to_fill):
  """
  strategy to what to do with NA in data

  df - DataFrame
  columns_to_fill - columns on which apply modification,
                    values with NA will be set to 0
  """
  df[columns_to_fill] = df[columns_to_fill].fillna(0)
  return df

def forward_fill_columns(df,columns_to_ffill):
  """
  strategy to what to do with NA in data

  df - DataFrame
  columns_to_ffill - columns on which apply modification,
                    values with NA will get previous found value which is not NA
  """
  df[columns_to_ffill] = df[columns_to_ffill].transform('ffill')
  return df

def create_merged_df(df_f, df_n, prefix):
    cols_to_keep=['Date', '^GSPC_Close', '^GSPC_Volume','EURUSD=X_Close',
    prefix+'_Open',prefix+ '_High',prefix+ '_Low',
    prefix+'_Close',prefix+ '_Volume']

    df_1 = df_f.loc[:,cols_to_keep]
    
    news_cols = df_n.drop(columns=["day"]).columns
    

    df_n['day'] = pd.to_datetime(df_n['day'])
    df_1['Date'] = pd.to_datetime(df_1['Date'])
    df_1 = df_1.transform('ffill')
    df_2 = pd.merge(df_n, df_1, left_on='day',right_on='Date', how='right')
        
    df_2[news_cols] = df_2[news_cols].fillna(0)
    df_2["mean_future"] = df_2[['future_finbert','future_bart','future_vader']].mean(axis=1)
    df_2["mean_influential"] = df_2[['influential_finbert','influential_bart','influential_vader']].mean(axis=1)
    df_2["mean_trustworthy"] = df_2[['trustworthy_finbert','trustworthy_bart','trustworthy_vader']].mean(axis=1)
    df_2["mean_clickbait"] = df_2[['clickbait_finbert','clickbait_bart','clickbait_vader']].mean(axis=1)

    df_2.drop(columns=["day"], inplace=True)
    return df_2

def calculate_technical_indicators(stock_df, date_col, open_col, high_col, low_col, close_col, volume_col, gspc_close,is_with_weekend=False):
    """
    Calculate additional technical indicators for a stock DataFrame.

    Parameters:
    - stock_df (pd.DataFrame): The input stock DataFrame.
    - date_col (str): Column name for date.
    - open_col (str): Column name for open prices.
    - high_col (str): Column name for high prices.
    - low_col (str): Column name for low prices.
    - close_col (str): Column name for close prices.
    - volume_col (str): Column name for volume.

    Returns:
    - pd.DataFrame: A new DataFrame containing calculated technical indicators.
    """
    # Combine the relevant columns into a DataFrame for ta library
    df = stock_df.loc[:,[date_col, open_col, high_col, low_col, close_col, volume_col,gspc_close]].copy()
    df.set_index(date_col, inplace=True)

    # Calculate technical indicators
    #df['sma_50'] = SMAIndicator(fillna=True,close=np.log(df[close_col]), window=50).sma_indicator()
    #df['ema_20'] = EMAIndicator(fillna=True,close=np.log(df[close_col]), window=20).ema_indicator()
    df['norm_rsi_14'] = RSIIndicator(fillna=True,close=df[close_col], window=14).rsi()/100
    
    df['norm_rsi_gspc_14'] = RSIIndicator(fillna=True,close=df[gspc_close], window=14).rsi()/100

    stoch = StochasticOscillator(fillna=True,close=df[close_col], low=df[low_col],high= df[high_col], window=14, smooth_window=3)
    df['norm_slowk_14'] = stoch.stoch()/100

    df['vwap_14'] = VolumeWeightedAveragePrice(fillna=True,high= df[high_col],low=df[low_col],close=df[close_col], volume=df[volume_col], window=14).volume_weighted_average_price()

    df['norm_roc_14'] =(ROCIndicator(df[close_col], window=14, fillna=True).roc() )/100
    
    if is_with_weekend:
        df['log_return_1'] = np.log(df[close_col]) - np.log(df[close_col]).shift(1)
        df['log_return_7'] = np.log(df[close_col]) - np.log(df[close_col]).shift(7)
        df['log_return_14'] = np.log(df[close_col]) - np.log(df[close_col]).shift(14)
        df['log_return_28'] = np.log(df[close_col]) - np.log(df[close_col]).shift(28)

        df['log_return_gspc_1'] = np.log(df[gspc_close]) - np.log(df[gspc_close]).shift(1)
        df['log_return_gspc_7'] = np.log(df[gspc_close]) - np.log(df[gspc_close]).shift(7)
        df['log_return_gspc_14'] = np.log(df[gspc_close]) - np.log(df[gspc_close]).shift(14)
        df['log_return_gspc_28'] = np.log(df[gspc_close]) - np.log(df[gspc_close]).shift(28)

    else:
        df['log_return_1'] = np.log(df[close_col]) - np.log(df[close_col]).shift(1)
        df['log_return_5'] = np.log(df[close_col]) - np.log(df[close_col]).shift(5)
        df['log_return_10'] = np.log(df[close_col]) - np.log(df[close_col]).shift(10)
        df['log_return_20'] = np.log(df[close_col]) - np.log(df[close_col]).shift(20)

        df['log_return_gspc_1'] = np.log(df[gspc_close]) - np.log(df[gspc_close]).shift(1)
        df['log_return_gspc_5'] = np.log(df[gspc_close]) - np.log(df[gspc_close]).shift(5)
        df['log_return_gspc_10'] = np.log(df[gspc_close]) - np.log(df[gspc_close]).shift(10)
        df['log_return_gspc_20'] = np.log(df[gspc_close]) - np.log(df[gspc_close]).shift(20)

    df['daily_variation'] = (df[high_col] - df[low_col]) / df[open_col]
    df['high_close_pressure'] = (df[high_col] - df[close_col]) / df[open_col]
    df['low_open_pressure'] = (df[low_col] - df[open_col]) / df[open_col]

    df['low_norm'] = df[low_col] / df['vwap_14']
    df['close_norm'] = df[close_col] / df['vwap_14']
    df['high_norm'] = df[high_col] / df['vwap_14']
    df['open_norm'] = df[open_col] / df['vwap_14']

    #merge
    stock_df_result = pd.merge(stock_df, df.reset_index().drop(columns=[open_col, high_col, low_col, close_col, volume_col, gspc_close]), on='Date', how='left',suffixes=('', '__duplicate'))

    return stock_df_result

def get_calculated_features(df, date_col, open_col, high_col, low_col, close_col, volume_col):
    """
    get features which are calculated

    df - DataFrame
    open_col, high_col, low_col, close_col, volume_col - columns in df
    """

    # Daily Variation
    df['Daily_Variation'] = (df[high_col] - df[low_col]) / df[open_col]

    # High — Close
    df['High_Close_Pressure'] = (df[high_col] - df[close_col]) / df[open_col]

    # Low — Open
    df['Low_Open_Pressure'] = (df[low_col] - df[open_col]) / df[open_col]

    # Amount
    df['Amount'] = df[volume_col] * df[close_col]

    # Price Change
    df['Price_Change'] = df[close_col]- df[close_col].shift(1)

    # # Weekday (weekday/4)
    # df[date_col] = pd.to_datetime(df[date_col])
    # df['weekday'] = df[date_col].dt.weekday/4

    # # Year Progress (month/12)
    # df['month'] = df[date_col].dt.month
    # df['Year_Progress'] = df['month'] / 12

    # # Month Progress (day/31)
    # df['day'] = df[date_col].dt.day
    # df['Month_Progress'] = np.round(df['day'] / 31, 2)

    # df.drop(columns=['month','day'], inplace=True)

    return df

def apply_weighted_weekend_news(df, date_col, weekend_news_cols, weights=None, debug=False):
    """
    strategy to what to do with NA in data

    df - DataFrame
    date_col - column in df with date
    weekend_news_cols - columns on which apply modification,
                        from friday to sunday there will be calculated mean on them
                        and put to friday index
    weights - weights for mean, in format for example [1,0,2], 1. number is weight
              for friday, 2. for saturday, 3. for sunday
    debug - if run function in debug mode, to see extra outputs
    """
    df[date_col] = pd.to_datetime(df[date_col])

    # Identify Fridays in the DataFrame using the 'Date' column
    fridays = df[df[date_col].dt.dayofweek == 4].index

    if debug == False:

    # Iterate over each Friday and apply the weighted mean of the weekend news
      for friday in fridays:
          if friday> df.shape[0]-2:
            break
          # Extract weekend news data for the specific weekend (from Friday to Sunday)
          weekend_news_data = df.loc[friday:friday +2, weekend_news_cols]
          # Calculate the weighted mean of the weekend news feature
          if weights is None:
              weekend_news_mean = weekend_news_data.mean()
          else:
              weighted_weekend_news = weekend_news_data.multiply(weights, axis=0)
              weekend_news_mean = weighted_weekend_news.sum(axis=0) / sum(weights)
          # Fill the Friday's columns with the calculated mean
          df.loc[friday, weekend_news_cols] = weekend_news_mean
    else:

      for friday in fridays:
          if friday> df.shape[0]-2:
            break
          weekend_news_data = df.loc[friday:friday +2, weekend_news_cols]
          print(weekend_news_data)

          if weights is None:
              weekend_news_mean = weekend_news_data.mean()
          else:
              print(weekend_news_data.multiply(weights, axis=0))
              weighted_weekend_news = weekend_news_data.multiply(weights, axis=0)
              weekend_news_mean = weighted_weekend_news.sum(axis=0) / sum(weights)

          print(weekend_news_mean)
          df.loc[friday, weekend_news_cols] = weekend_news_mean

    df = df[df[date_col].dt.dayofweek < 5]  # Keep only weekdays
    return df

def min_max_scale(df_to_scale,date_col, columns_to_scale, train_data=True, scaler=None):
    """
    scale specified columns using Min-Max scaling.

    df - DataFrame
    columns_to_scale - list of column names to be scaled
    train_data - bool, if True, perform scaling and return both the scaled DataFrame and the scaler
                 if False, use the provided scaler to scale the DataFrame
    scaler - sklearn.preprocessing.MinMaxScaler or None,if train_data is False,
            provide a pre-fitted scaler to use for scaling

    """
    df = df_to_scale.loc[:,columns_to_scale+[date_col]].copy()
    df.set_index(date_col, inplace=True)

    if train_data:
        # Perform Min-Max scaling and return both the scaled DataFrame and the scaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=['minmax_'+col for col in columns_to_scale], index=df.index)

        scaled_df = pd.merge(df_to_scale.drop(columns=columns_to_scale), scaled_df.reset_index(), on='Date', how='left',suffixes=('', '__duplicate'))

        return scaled_df, scaler
    elif scaler is not None:
        # Use the provided scaler to scale the DataFrames
        scaled_data = scaler.transform(df[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=['minmax_'+col for col in columns_to_scale], index=df.index)

        scaled_df = pd.merge(df_to_scale.drop(columns=columns_to_scale), scaled_df.reset_index(), on='Date', how='left',suffixes=('', '__duplicate'))

        return scaled_df
    else:
        raise ValueError("If train_data is False, a pre-fitted scaler must be provided.")