import datetime
import requests
import pandas as pd
import re
import os


class AlphaVantageNewsDownloader:
    """
    Class for downloading news data from AlphaVantage. API key(s) are required.
    """

    def __init__(self, api_keys: list[str], ticker: str, begin_date: str, end_date: str, days_per_request: int = 30) -> None:
        """
        :param api_keys: list[str], list of API keys
        :param ticker: str, ticker to download news for (i.e. AAPL)
        :param begin_date: str, begin date in format YYYYMMDD
        :param end_date: str, end date in format
        :param days_per_request: int, number of days per request (optimal is 30)
        """
        self.api_keys = api_keys
        self.ticker = ticker
        self.begin_date = datetime.datetime.strptime(begin_date, "%Y%m%d")
        self.end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
        if self.end_date <= self.begin_date:
            raise ValueError("End date must be greater than begin date")
        if self.end_date > datetime.datetime.now():
            raise ValueError("End date must be less or equal than current date")
        self.days_per_request = days_per_request


    def download_raw_news_data(self, begin_date: datetime.date, end_date: datetime.date, limit: int = 1000) -> dict:
        """
        Downloads raw news data from AlphaVantage.

        :param begin_date: datetime.date, begin date
        :param end_date: datetime.date, end date
        :param limit: int, limit of news per request (max is 1000)
        :return: dict, raw news data
        """
        if limit > 1000:
            raise(ValueError("AlphaVantage supports news limit up to 1000 per request"))

        begin_date_f = datetime.datetime.strftime(begin_date, "%Y%m%d")
        end_date_f = datetime.datetime.strftime(end_date, "%Y%m%d")

        for api_key in self.api_keys:
            print(f"Downloading raw news from {begin_date_f} to {end_date_f}...")
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&time_from={begin_date_f}T0000&time_to={end_date_f}T0000&limit={limit}&apikey={api_key}&datatype=csv"
            r = requests.get(url)
        
            data_news = r.json()
            if len(data_news)>1:
                print("Raw news downloaded correctly")
                return data_news
            print("WARNING: Raw news not downloaded correctly, you might have exceeded API limit")
        return {}


    def convert_raw_news_data(self, data_news: dict):
        """
        Converts raw news data to a dictionary that can be easily converted to a dataframe.

        :param data_news: dict, raw news data
        :return: dict, converted news data
        """

        dict_news = {"title":[], "url":[], "summary":[], "source": [], "topics": [], 
                     "category_within_source": [], "authors": [], "overall_sentiment_score":[],
                     "overall_sentiment_label":[], "ticker_relevance_score":[],
                     "ticker_sentiment_score":[], "ticker_sentiment_label":[],
                     "time_published":[]}
        
        ticker_sentiment_keys = ["ticker_relevance_score", "ticker_sentiment_score", "ticker_sentiment_label"]

        for event in data_news.get("feed", []):
            for key in dict_news:
                if key not in ticker_sentiment_keys:
                    dict_news[key].append(event.get(key, None))

            for ticker in event["ticker_sentiment"]:
                if ticker["ticker"]==self.ticker:
                    for key in ticker_sentiment_keys:
                        dict_news[key].append(ticker.get(key, None))
        print(f"Number of added news: {len(data_news.get('feed', []))}")
        return dict_news
    
    def download_multiple_data(self):
        """
        Downloads data for multiple dates from AlphaVantage.

        :return: dict, converted news data
        """
        dict_news = {"title":[], "url":[], "summary":[], "overall_sentiment_score":[],
                "overall_sentiment_label":[], "ticker_relevance_score":[],
                "ticker_sentiment_score":[], "ticker_sentiment_label":[],
                "time_published":[]}
        current_date = self.begin_date

        for _ in range(5):
            if current_date >= self.end_date:
                print("End date reached")
                return dict_news
            
            cur_end_date = current_date + datetime.timedelta(days=self.days_per_request)

            if cur_end_date >= self.end_date:
                cur_end_date = self.end_date
            
            data_news = self.download_raw_news_data(current_date, cur_end_date)
            if data_news=={}:
                self.end_date = current_date
                return dict_news
            dict_news_tmp = self.convert_raw_news_data(data_news)
            for key in dict_news:
                dict_news[key] += dict_news_tmp[key]
            
            current_date = cur_end_date
        self.end_date = current_date
        return dict_news

        # while current_date < self.end_date:
        #     data_news = self.download_raw_news_data(current_date, current_date + datetime.timedelta(days=self.days_per_request))
        #     if data_news=={}:
        #         self.end_date = current_date
        #         return dict_news
        #     dict_news_tmp = self.convert_raw_news_data(data_news)
        #     for key in dict_news:
        #         dict_news[key] += dict_news_tmp[key]

        #     current_date += datetime.timedelta(days=self.days_per_request)
        # return dict_news
    

    def save_to_dir(self, dict_news: dict) -> None:
        """
        Saves news data to a directory.

        :param dict_news: dict, news data
        """
        print("Saving...")
        news_path = os.path.join("DATA", "alphavantage", "news")
        dir_path = os.path.join(news_path, re.sub(r'[^A-Za-z0-9]+', '', self.ticker))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = os.path.join("DATA", "alphavantage", "news", re.sub(r'[^A-Za-z0-9]+', '', self.ticker), f"{re.sub(r'[^A-Za-z0-9]+', '', self.ticker)}_{datetime.datetime.strftime(self.begin_date, '%Y%m%d')}_{datetime.datetime.strftime(self.end_date, '%Y%m%d')}.csv")
        pd.DataFrame(dict_news).to_csv(path, index=False)
        print(f"Saved file to path: {path}")

def main():
    # This is just usage example, not part of the class
    # You can use only one api_key, but there is a limit of 5 requests per minute, so it might be helpful to use more

    # WARNING!!!: News Data is available only from 01.03.2022 - cannot use data before that
    api_keys = ["BC1SIZ29L8F77M2A"]
    ticker = "BA"
    begin_date = "20210101"
    end_date = "20210131"
    days_per_request = 21


    avnd = AlphaVantageNewsDownloader(api_keys, ticker, begin_date, end_date, days_per_request)
    dict_news = avnd.download_multiple_data()

    # dict_news = avnd.download_raw_news_data(avnd.begin_date, avnd.end_date)
    # dict_news = avnd.convert_raw_news_data(dict_news)
    avnd.save_to_dir(dict_news)
    
if __name__ == "__main__":
    main()