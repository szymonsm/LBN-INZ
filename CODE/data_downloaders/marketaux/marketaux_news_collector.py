import datetime
import requests
import pandas as pd
import re
import os
from collections import defaultdict, Counter
from datetime import timedelta
import sys

TICKER_TICKER_NAME = {"AAPL": "Apple", "TSLA": "Tesla", "BA": "Boeing", "NFLX": "Netflix", "BTCUSD": "Bitcoin USD"}


class MarketauxNewsDownloader:
    """
    Class for downloading news data from AlphaVantage. API key(s) are required.
    """

    def __init__(self, api_keys: list[str], ticker_name: str, begin_date: str, end_date: str) -> None:
        """
        :param api_keys: list[str], list of API keys
        :param ticker_name: str, ticker name to download news for (i.e. Apple, Tesla, etc.)
        :param begin_date: str, begin date in format YYYY-MM-DD
        :param end_date: str, end date in format YYYY-MM-DD
        """
        self.api_keys = api_keys
        self.main_api_key_idx = 0
        self.ticker_name = ticker_name
        self.begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
        self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        if self.end_date <= self.begin_date:
            raise ValueError("End date must be greater than begin date")
        if self.end_date > datetime.datetime.now():
            raise ValueError("End date must be less or equal than current date")


    def download_raw_news_data(self, begin_date: datetime.date, page: int, sort_key: str = "entity_match_score") -> dict:
        """
        Downloads raw news data from marketaux.

        :param begin_date: datetime.date, begin date
        :param page: int, page number
        :param sort_key: str, sort key
        :return: dict, raw news data
        """

        #for api_key in self.api_keys:
        str_date = datetime.datetime.strftime(begin_date, "%Y-%m-%d")
        print(f"Downloading raw news from {str_date}...")
        url = f"https://api.marketaux.com/v1/news/all?symbols={self.ticker_name}&language=en&page={page}&published_on={str_date}&sort={sort_key}&api_token={self.api_keys[self.main_api_key_idx]}"
        r = requests.get(url)
    
        data_news = r.json()
        if len(data_news.get("data", []))>0:
            print("Raw news downloaded correctly")
            return data_news
        if "error" in data_news and data_news["error"]["code"]=="rate_limit_reached":
            self.main_api_key_idx += 1
            if self.main_api_key_idx>=len(self.api_keys):
                #sys.exit("All API keys exceeded rate limit")
                raise ValueError(f"All API keys exceeded rate limit. Last saved date: {str_date}.")
            else:
                print(f"New api key used: {self.api_keys[self.main_api_key_idx]}")
                return self.download_raw_news_data(begin_date, page, sort_key)
            # print("WARNING: Raw news not downloaded correctly, you might have exceeded API limit")
            # return {}
        #print(data_news)
        print("WARNING: Probably day without news.")
        return {}


    def convert_raw_news_data(self, data_news: dict):
        """
        Converts raw news data to a dictionary that can be easily converted to a dataframe.

        :param data_news: dict, raw news data
        :return: dict, converted news data
        """

        dict_news = defaultdict(list)

        for event in data_news["data"]:
            for key in event:
                if isinstance(event[key], str):
                    dict_news[key].append(event[key])
                elif key=="relevance_score":
                    dict_news["relevance_score"].append(event["relevance_score"])
                elif key=="entities":
                    entity_dict = defaultdict(list)
                    for entity in event["entities"]:
                        if TICKER_TICKER_NAME[self.ticker_name].lower() in entity["name"].lower():
                            entity_dict["type"].append(entity["type"])
                            entity_dict["industry"].append(entity["industry"])
                            entity_dict["match_score"].append(entity["match_score"])
                            entity_dict["sentiment_score"].append(entity["sentiment_score"])

                    for keyy in entity_dict:   
                        if keyy!="type" and keyy!="industry":
                            dict_news[keyy].append(sum(entity_dict[keyy])/len(entity_dict[keyy]))
                        else:
                            c = Counter(entity_dict[keyy])
                            dict_news[keyy].append(c.most_common(1)[0][0] or -1)
        print("Number of added news: ", len(dict_news["title"]))
        return dict_news
    
    def download_multiple_data(self, max_num_requests: int = 10, pages: int | list[int] = [1]):
        """
        Downloads data for multiple dates from AlphaVantage.

        :return: dict, converted news data
        """
        dict_news = {"uuid": [], "title":[], "description":[], "keywords": [], 
                     "snippet": [], "url": [], "image_url": [], "language": [], 
                     "published_at": [], "source": [], "relevance_score": [], 
                     "type": [], "industry": [], "match_score": [], "sentiment_score": []}
        if isinstance(pages, int):
            pages = [pages]

        current_date = self.begin_date
        i = 0 

        while i<max_num_requests:
            if current_date > self.end_date:
                print("End date reached")
                return dict_news
            
            for page in pages:
                data_news = self.download_raw_news_data(current_date, page)
                if data_news=={}:
                    self.end_date = current_date
                    return dict_news
                dict_news_tmp = self.convert_raw_news_data(data_news)
                for key in dict_news:
                    dict_news[key] += dict_news_tmp[key]
                i += 1
                if i>=max_num_requests:
                    break
                if len(dict_news_tmp["title"]) <3:
                    break

            current_date += datetime.timedelta(days=1)
        self.end_date = min(self.end_date, current_date)
        return dict_news
 

    def save_to_dir(self, dict_news: dict) -> None:
        """
        Saves news data to a directory.

        :param dict_news: dict, news data
        """
        print("Saving...")
        news_path = os.path.join("DATA", "marketaux", "news")
        dir_path = os.path.join(news_path, re.sub(r'[^A-Za-z0-9]+', '', self.ticker_name))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = os.path.join("DATA", "marketaux", "news", re.sub(r'[^A-Za-z0-9]+', '', self.ticker_name), f"{re.sub(r'[^A-Za-z0-9]+', '', self.ticker_name)}_{datetime.datetime.strftime(self.begin_date, '%Y%m%d')}_{datetime.datetime.strftime(self.end_date, '%Y%m%d')}.csv")
        pd.DataFrame(dict_news).to_csv(path, index=False)
        print(f"Saved file to path: {path}")

def main():
    # This is just usage example, not part of the class
    # You can use only one api_key, but there is a limit of 100 requests per day, so it might be helpful to use more
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
    ticker_name = "BTCUSD"
    begin_date = "2023-05-07"
    end_date = "2023-11-26"

    downloader = MarketauxNewsDownloader(api_keys, ticker_name, begin_date, end_date)
    dict_news = downloader.download_multiple_data(max_num_requests=100, pages=[1, 2, 3])
    downloader.save_to_dir(dict_news)
    while downloader.end_date < datetime.datetime.strptime(end_date, "%Y-%m-%d"):
        downloader.begin_date = downloader.end_date + timedelta(days=1)
        downloader.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        dict_news = downloader.download_multiple_data(max_num_requests=100, pages=[1, 2, 3])
        downloader.save_to_dir(dict_news)

    
if __name__ == "__main__":
    main()