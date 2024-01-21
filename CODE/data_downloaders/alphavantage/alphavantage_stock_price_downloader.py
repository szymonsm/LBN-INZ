import pandas as pd
import os
import re
import datetime

ALLOWED_INTERVALS = [1, 5, 15, 30, 60]

class AlphaVantageStockPriceDownloader:
    """
    Class for downloading stock price data from AlphaVantage. API key(s) are required.
    """
    def __init__(self, api_keys: list[str], ticker: str):
        """
        :param api_keys: list[str], list of API keys
        :param ticker: str, ticker to download news for (i.e. AAPL)
        """
        self.api_keys = api_keys
        self.ticker = ticker


    def download_daily_ticker_data(self) -> pd.DataFrame | None:
        """
        Downloads historical daily ticker data for 20+ years from AlphaVantage.

        :return: pd.DataFrame, daily ticker data or None if no data was downloaded
        """

        for api_key in self.api_keys:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.ticker}&outputsize=full&apikey={api_key}&datatype=csv'
            df_daily = pd.read_csv(url)
            print(f"Number of downloaded rows: {len(df_daily)}")
            if len(df_daily)>2:
                return df_daily
        return None 
    
    def download_intraday_ticker_data(self, month: str, interval: int) -> pd.DataFrame | None:
        """
        Downloads historical intraday ticker data for a given month and time interval from AlphaVantage.

        :param month: str, month in format YYYY-MM
        :param interval: int, interval in minutes (1, 5, 15, 30, 60)
        :return: pd.DataFrame, intraday ticker data or None if no data was downloaded
        """
        if interval not in ALLOWED_INTERVALS:
            raise ValueError(f"Given interval is not available, choose from this list: {ALLOWED_INTERVALS}")
        for api_key in self.api_keys:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.ticker}&interval={interval}min&month={month}&outputsize=full&apikey={api_key}&datatype=csv'
            df_daily = pd.read_csv(url)
            print(f"Number of downloaded rows: {len(df_daily)}")
            if len(df_daily)>2: 
                return df_daily
        return None 
    
    def save_to_dir(self, df_price: pd.DataFrame, type: str = "daily", month: str | None = None, interval: int | None = None) -> None:
        """
        Saves price data to a directory.

        :param df_price: pd.DataFrame, price data
        :param type: str, type of data (daily or intraday)
        :param month: str, month in format YYYY-MM
        :param interval: int, interval in minutes (1, 5, 15, 30, 60)
        """
        assert type in ["daily", "intraday"], "Type must be either daily or intraday"
        if type == "intraday":
            assert month is not None, "Month must be specified for intraday data"
            assert interval is not None, "Interval must be specified for intraday data"

        print("Saving...")
        path = os.path.join("DATA", "alphavantage", type)
        dir_path = os.path.join(path, re.sub(r'[^A-Za-z0-9]+', '', self.ticker))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if type == "intraday":
            path = os.path.join(dir_path, f"{re.sub(r'[^A-Za-z0-9]+', '', self.ticker)}_intraday_{interval}_{datetime.datetime.strftime(month, '%Y%m%d')}.csv")
        else:
            path = os.path.join(dir_path, f"{re.sub(r'[^A-Za-z0-9]+', '', self.ticker)}_daily_full.csv")
        df_price.to_csv(path, index=False)
        print(f"Saved file to path: {path}")


def main():
    # This is just usage example, not part of the class
    api_keys = ["BC1SIZ29L8F77M2A"]
    # ticker = "BA"
    # months = ["2023-0"+str(i) for i in range(1,10)]
    # interval = 15

    # avspd = AlphaVantageStockPriceDownloader(api_keys, ticker)
    # for month in months:
    #     df = avspd.download_intraday_ticker_data(month, interval)
    #     if df is not None:
    #         df.to_csv(f"DATA/alphavantage/intraday/{ticker}/{ticker}_intraday_{interval}_{month}.csv", index=False)

    ticker = "NFLX"
    avspd = AlphaVantageStockPriceDownloader(api_keys, ticker)
    df = avspd.download_daily_ticker_data()
    df.to_csv(f"DATA/alphavantage/daily/{ticker}_daily_full.csv", index=False)

    # df = avspd.download_daily_ticker_data()
    # df.to_csv(f"{ticker}_daily_full.csv")
if __name__ == "__main__":
    main()
