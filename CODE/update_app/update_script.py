from code.news_collector import MarketauxNewsDownloader
from code.yahoofinanse_downloader import YFinanceDownloader
from code.news_pipeline import NewsSentimentProcessor
from code.scripts.essentials import *
import pandas as pd
from datetime import datetime, timedelta
import os
import git
from datetime import datetime
import pickle
from code.scripts.train_utilities import window_dataset
from keras.models import load_model

symbols = ['BA', 'BTC-USD', 'NFLX', 'TSLA']
additional_symbols = ['^GSPC', 'EURUSD=X']
ticker_names = ['BA', 'BTCUSD', 'NFLX', 'TSLA']

api_keys = ["VhQlodkXbHbZE0vUS6eQhYwdSvfgQSP4NjLrVACo",
            "B12FwLhv80TqecMLNRdaiMIZiTPOKzhxMY2tLwWY",
            "m0Q3p7snIwLxxVueQjY2QkORIUKvMrDB4CLy7xi8",
            "65tlBN9OAjlbR2EGMOb97BnTtf1axhOUw8YFSiZZ",
            "6IGEFIcvpdJiSdmTH99BSn8AVfftuOnHAwI8vkZX",
            "07O7pQpMO2hruWQmU6S8LJncJrD9tYoQEa94mjL7",
            "7jpIvaNVSfbzKgSjBJUjKLGOkheDqlXXHow0hqV2",
            "ru9csB3TBwCjpfyUzw4f3hE2vB88y8iiER0dpXj3",
            "egwiukhKomAbje2lrrKL1z1ECN8ienRm6aBd5Wsp",
            "9oxxLUTAiMSXceql18HZ4Xz1CKhwa2bVtEMuqIAQ",
            "UuxWMNP8XfDuppbNinpvBYOfedJouSYMEwjEUwOq",
            "gk0849Cv6DTfsMAmzp13qp7Q09ULfgnNWHd8R3LS",
            "TEDETqt2eka8HbEXr9iHafaQaaC8yntxJ4kqg5DV",
            "1A6BIomNHkYfwuVEXkS06zz2CmVRtOYKy451cj8k",
            "DklBEXK6POj73EGZuTAYMN9EEHBLMWe9Flgenz0P",
            "TXIDyS8bYmUYUXLkSZIJ27HMPn4G1XrjxmgRONYk",
            "d01yoYY7lebNcRsFB4jLWLJoYsO1ANSay1frZeCc",
            "FOlRHekaaKXDO8OzQCm3OhFXcn9AN87InF6poZye",
            "SbVF63j1kq1oLd8xLrXb1c9NePopE78XlqukJLlR", 
            "n7Z3u8RoupEccoFTSilWh9S8XpTvFVJidSZS7rfR",
            'hDxiXlgvgd5SBo0zzSWVYIJdnFk4r7Bfjc4N62du',
            'VXaKyxfn2nFD0SXV7wmW1BB9eagIdhw47plxumag',
            'mRmoOIAtcd862pkbbON8eospdaKlyN6i0GQikHXa',
            'fhTKQOtkv136elD1O4wE87giBKHJKjrOzoLiGuzv',
            'RspkY5dUGKD16wDJytHjHNv6sVbRTulvcyugRDYU',
            'SnzD2thuyDWGZ4dg5OOKzRTIS1rAFhXY2Hzd1xjH', 
            'rRIg0OrTma5LQR9a4zF37clVAAyUB9ZUaAupdJgL']

def update_news(symbols, ticker_names, api_keys, update_github=True):
    """
    Updates news data for given symbols, ticker names with given api keys.
    Then commits and pushes changes to github repository.

    :param symbols: list, list of symbols to update
    :param ticker_names: list, list of ticker names to update
    :param api_keys: list, list of api keys used to update data with MarketauxNewsDownloader

    :return: 1 if update was successful, -1 if there was no new data to download

    """
    
    r = 1

    for symbol, ticker_name in zip(symbols, ticker_names):

        print(f"Dowloading {symbol} news data...")
        df = pd.read_csv(f'{symbol}/raw_news.csv')
        begin_date = (pd.to_datetime(df['published_at']).dt.date + pd.DateOffset(days=1)).max().strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        if begin_date > end_date:
            print('No new data to download')
            r = -1
            continue
        downloader = MarketauxNewsDownloader(api_keys, ticker_name, begin_date, end_date)
        dict_news = downloader.download_multiple_data(max_num_requests=100, pages=[1, 2, 3])
        df_to_append = pd.DataFrame(dict_news)
        while downloader.end_date < datetime.strptime(end_date, "%Y-%m-%d"):
            downloader.begin_date = downloader.end_date + timedelta(days=1)
            downloader.end_date = datetime.strptime(end_date, "%Y-%m-%d")
            dict_news = downloader.download_multiple_data(max_num_requests=100, pages=[1, 2, 3])
            df_to_append = pd.concat([df_to_append, pd.DataFrame(dict_news)]).reset_index(drop=True)
        df = pd.concat([df, df_to_append]).reset_index(drop=True)
        df.to_csv(f'{symbol}/raw_news.csv', index=False)
        if update_github:
            repo = git.Repo(os.getcwd(), search_parent_directories=True)
            repo.git.add(update=True)
            repo.index.commit(f'Update {symbol} news {datetime.today().strftime("%Y-%m-%d")}')
            origin = repo.remote(name='origin')
            origin.push()
        print(f"Downloaded {symbol} news data.")
    return r

def update_finance(symbols, update_github=True):

    """
    Updates finance data for given symbols.
    Then commits and pushes changes to github repository.

    :param symbols: list, list of symbols to update

    :return: 1 if update was successful, -1 if there was no new data to download

    """
    r = 1

    for symbol in symbols:
        print(f"Dowloading {symbol} finance data...")
        df = pd.read_csv(f'{symbol}/raw_finance.csv')

        begin_date = (pd.to_datetime(df['Date']).dt.date + pd.DateOffset(days=1)).max().strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        if begin_date > end_date:
            print('No new data to download')
            r = -1
            continue
    
        begin_date = begin_date.replace('-', '')
        end_date = end_date.replace('-', '')
        symbol_list = [symbol]
        yfd = YFinanceDownloader(symbol_list,begin_date,end_date)
        df_ = yfd.create_df()
        df = pd.concat([df, df_]).reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df.to_csv(f'{symbol}/raw_finance.csv', index=False)
        if update_github:
            repo = git.Repo(os.getcwd(), search_parent_directories=True)
            repo.git.add(update=True)
            repo.index.commit(f'Update {symbol} finance {datetime.today().strftime("%Y-%m-%d")}')
            origin = repo.remote(name='origin')
            origin.push()
        print(f"Downloaded {symbol} finance data.")
    return r

def update_processed(symbols, update_github=True):
    """
    Updates processed data for given symbols.
    Then commits and pushes changes to github repository.

    :param symbols: list, list of symbols to update

    :return: 1 if update was successful, -1 if there was no new data to process
    """

    r = 1

    for symbol in symbols:

        df = pd.read_csv(f'{symbol}/processed.csv')
        df_news = pd.read_csv(f'{symbol}/raw_news.csv')

        begin_date = (pd.to_datetime(df['Date']).dt.date + pd.DateOffset(days=1)).max().strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        if begin_date > end_date:
            print('No new data to process')
            r = -1
            continue
        df_news = df_news[(df_news['published_at'] >= begin_date) & (df_news['published_at'] <= end_date)]
        df_news = df_news.reset_index(drop=True)

        news_sentiment_processor = NewsSentimentProcessor(symbol, df_news, device=-1)
        df_result = news_sentiment_processor.launch()

        df_finance = pd.read_csv(symbol+"/raw_finance.csv")
        df_finance_gspc = pd.read_csv("^GSPC/raw_finance.csv")
        df_finance_eur = pd.read_csv("EURUSD=X/raw_finance.csv")
        df_finance = df_finance[(df_finance['Date'] >= begin_date) & (df_finance['Date'] <= end_date)].reset_index(drop=True)
        df_finance_gspc = df_finance_gspc[(df_finance_gspc['Date'] >= begin_date) & (df_finance_gspc['Date'] <= end_date)].drop(['is_weekend'], axis=1).reset_index(drop=True)
        df_finance_eur = df_finance_eur[(df_finance_eur['Date'] >= begin_date) & (df_finance_eur['Date'] <= end_date)].drop(['is_weekend'], axis=1).reset_index(drop=True)
        
        df_finance = df_finance.merge(df_finance_gspc, on='Date', how='left')
        df_finance = df_finance.merge(df_finance_eur, on='Date', how='left')

        df_m = create_merged_df(df_finance, df_result, symbol)

        if symbol != 'BTC-USD':
            cols_news = ['future', 'influential', 'trustworthy', 'not clickbait',
                        'finbert_Score', 'bart_Score', 'vader_Score', 'future_finbert',
                        'future_bart', 'future_vader', 'influential_finbert',
                        'influential_bart', 'influential_vader', 'trustworthy_finbert',
                        'trustworthy_bart', 'trustworthy_vader', 'clickbait_finbert',
                        'clickbait_bart', 'clickbait_vader',
                        'mean_future','mean_influential', 'mean_trustworthy', 'mean_clickbait']
            df_processed = apply_weighted_weekend_news(df_m,'Date', cols_news, weights=[0.6,0.2,0.2])
            df_processed = calculate_technical_indicators(df_processed,'Date',symbol+'_Open',symbol+ '_High',symbol+ '_Low', symbol+'_Close',symbol+ '_Volume','^GSPC_Close')
            df_processed['target_1'] = df_processed['log_return_1'].shift(-1)
            df_processed['target_5'] = df_processed['log_return_5'].shift(-5)
            df_processed['target_10'] = df_processed['log_return_10'].shift(-10)
            df_processed['target_20'] = df_processed['log_return_20'].shift(-20)
        else:
            df_processed = calculate_technical_indicators(df_m,'Date',symbol+'_Open',symbol+ '_High',symbol+ '_Low', symbol+'_Close',symbol+ '_Volume','^GSPC_Close',True)
            df_processed['target_1'] = df_processed['log_return_1'].shift(-1)
            df_processed['target_7'] = df_processed['log_return_7'].shift(-7)
            df_processed['target_14'] = df_processed['log_return_14'].shift(-14)
            df_processed['target_28'] = df_processed['log_return_28'].shift(-28)

        cols_min_max = ['^GSPC_Volume', symbol+'_Volume',
                        'daily_variation', 'high_close_pressure', 'low_open_pressure',
                        'low_norm', 'close_norm', 'high_norm', 'open_norm']
        with open(f'scalers/scaler_min_max_{symbol}.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
        output = min_max_scale(df_processed,'Date', cols_min_max, train_data=False, scaler=loaded_scaler)

        df_old = pd.read_csv(f'{symbol}/processed.csv')
        df_new = pd.concat([df_old, output]).reset_index(drop=True)
        df_new['Date'] = pd.to_datetime(df_new['Date']).dt.date
        df_new.to_csv(f'{symbol}/processed.csv', index=False)

        if update_github:
            repo = git.Repo(os.getcwd(), search_parent_directories=True)
            repo.git.add(update=True)
            repo.index.commit(f'Update {symbol} processed {datetime.today().strftime("%Y-%m-%d")}')
            origin = repo.remote(name='origin')
            origin.push()
    return r

def update_predictions(symbols, update_github=True):

    """
    Updates predictions for given symbols.
    Then commits and pushes changes to github repository.

    :param symbols: list, list of symbols to update

    :return: 1 if update was successful
    """

    for symbol in symbols:

        data = pd.read_csv(f"{symbol}/processed.csv")

        if symbol == "BA":
            window_size = 10
            target_cols = ['target_5']
            cols_used = [
                    'norm_rsi_14', 'norm_slowk_14', 'minmax_daily_variation', 'minmax_BA_Volume',
                    'mean_influential', 'mean_trustworthy', 'finbert_Score', 'bart_Score'
                    ]
            model = load_model('models/'+symbol+'_lstm_full_cols.h5', compile=False)

            step = 5

        elif symbol == "TSLA":
            window_size = 10
            target_cols = ['target_5']
            cols_used = [
                        'minmax_low_norm', 'minmax_high_norm', 'norm_rsi_gspc_14', 'norm_slowk_14' ,
                        'vader_Score', 'bart_Score', 'mean_influential', 'finbert_Score', 'mean_trustworthy'
                        ]
            model = load_model('models/'+symbol+'_lstm_full_cols.h5', compile=False)
            step = 5

        elif symbol == "NFLX":

            window_size = 10
            target_cols = ['target_5']
            cols_used = [
                        'norm_rsi_gspc_14', 'norm_rsi_14',
                        'norm_slowk_14', 'minmax_high_norm', 'log_return_1'
                        ]
            model = load_model('models/'+symbol+'_lstm_fin_cols.h5', compile=False)
            step = 5

        elif symbol == "BTC-USD":
            window_size = 14
            target_cols = ['target_7']
            cols_used = [
                        'minmax_BTC-USD_Volume', 'norm_rsi_14', 'norm_slowk_14', 'norm_rsi_gspc_14', 'minmax_daily_variation',
                        'finbert_Score', 'vader_Score', 'mean_influential'
                        ]
            model = load_model('models/'+symbol+'_lstm_full_cols.h5', compile=False)
            step = 7

        else:
            print("Prefix not found")
            exit()

        X, _ = window_dataset(data[list(cols_used)+target_cols], target_cols[0], window_size)
        y_pred = model.predict(X)
        dates = data['Date']

        if step == 5:

            next_dates = list(pd.bdate_range(start=dates.max(), periods=step+1)[1:].strftime('%Y-%m-%d'))
            dates_extended = list(dates)+next_dates
        
        else:

            next_dates = list(pd.date_range(start=dates.max(), periods=step+1)[1:].strftime('%Y-%m-%d'))
            dates_extended = list(dates)+next_dates        

        df_pred = pd.DataFrame({"Prediction":y_pred.flatten(), "Date": dates_extended[(window_size+step):]})
        df_pred.to_csv(f'{symbol}/predictions.csv', index=False)

        if update_github:
            repo = git.Repo(os.getcwd(), search_parent_directories=True)
            repo.git.add(update=True)
            repo.index.commit(f'Update {symbol} predictions {datetime.today().strftime("%Y-%m-%d")}')
            origin = repo.remote(name='origin')
            origin.push()

    return 1
    
def main():

    update_news(symbols, ticker_names, api_keys)
    update_finance(symbols)
    update_finance(additional_symbols, update_github=False)
    update_processed(symbols)
    update_predictions(symbols)

if __name__ == '__main__':
    main()