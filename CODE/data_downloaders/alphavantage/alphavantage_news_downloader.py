import datetime
import requests
import pandas as pd
import re


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
        self.days_per_request = days_per_request


    def download_raw_news_data(self, begin_date: datetime.date, end_date: datetime.date, limit: int = 1000) -> dict:
        """
        Downloads raw news data from AlphaVantage.

        :param begin_date: datetime.date, begin date
        :param end_date: datetime.date, end date
        :param limit: int, limit of news per request (max is 1000)
        :return: dict, raw news data
        """

        assert limit<=1000, "AlphaVantage supports news limit up to 1000 per request"
        for api_key in self.api_keys:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&time_from={datetime.datetime.strftime(begin_date, '%Y%m%d')}T0000&time_to={datetime.datetime.strftime(end_date, '%Y%m%d')}T0000&limit={limit}&apikey={api_key}&datatype=csv"
            r = requests.get(url)
        
            data_news = r.json()
            if len(data_news)>1:
                return data_news
        return data_news


    def convert_raw_news_data(self, data_news: dict):
        """
        Converts raw news data to a dictionary that can be easily converted to a dataframe.

        :param data_news: dict, raw news data
        :return: dict, converted news data
        """

        dict_news = {"title":[], "url":[], "summary":[], "overall_sentiment_score":[],
                     "overall_sentiment_label":[], "ticker_relevance_score":[],
                     "ticker_sentiment_score":[], "ticker_sentiment_label":[],
                     "time_published":[]}

        for event in data_news.get("feed", []):
            dict_news["title"].append(event["title"])
            dict_news["url"].append(event["url"])
            dict_news["summary"].append(event["summary"])
            dict_news["source"].append(event["source"])
            dict_news["topics"].append(event["topics"])
            dict_news["category_within_source"].append(event["category_within_source"])
            dict_news["authors"].append(event["authors"])
            dict_news["overall_sentiment_score"].append(event["overall_sentiment_score"])
            dict_news["overall_sentiment_label"].append(event["overall_sentiment_label"])
            dict_news["time_published"].append(event["time_published"])

            for ticker in event["ticker_sentiment"]:
                if ticker["ticker"]==self.ticker:
                    dict_news["ticker_relevance_score"].append(ticker["relevance_score"])
                    dict_news["ticker_sentiment_score"].append(ticker["ticker_sentiment_score"])
                    dict_news["ticker_sentiment_label"].append(ticker["ticker_sentiment_label"])
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
        while current_date <= self.end_date:
            data_news = self.download_raw_news_data(current_date, current_date + datetime.timedelta(days=self.days_per_request))
            dict_news_tmp = self.convert_raw_news_data(data_news)
            for key in dict_news:
                dict_news[key] += dict_news_tmp[key]

            current_date += datetime.timedelta(days=self.days_per_request)
        return dict_news
    

def main():
    # This is just usage example, not part of the class
    # You can use only one api_key, but there is a limit of 5 requests per minute, so it might be helpful to use more
    api_keys = ["YOUR_API_KEY_1", "YOUR_API_KEY_2"]
    ticker = "CRYPTO:BTC"
    begin_date = "20230101"
    end_date = "20231001"
    days_per_request = 30


    avnd = AlphaVantageNewsDownloader(api_keys, ticker, begin_date, end_date, days_per_request)
    dict_news = avnd.download_multiple_data()
    pd.DataFrame(dict_news).to_csv(f"data/daily{re.sub(r'[^A-Za-z0-9]+', '', avnd.ticker)}_{begin_date}_{end_date}.csv", index=False)
    
if __name__ == "__main__":
    main()