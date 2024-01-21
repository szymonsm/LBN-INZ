from CODE.data_downloaders.alphavantage.alphavantage_news_downloader import AlphaVantageNewsDownloader
from CODE.data_downloaders.alphavantage.alphavantage_stock_price_downloader import AlphaVantageStockPriceDownloader
from CODE.data_downloaders.yahoofinanse.yahoofinanse_downloader import YFinanceDownloader
from CODE.model_training.scripts.essentials import calculate_technical_indicators

from unittest import TestCase
from datetime import date, timedelta, datetime

class AlphaVantageNewsDownloaderTest(TestCase):

    avnd = AlphaVantageNewsDownloader(["BC1SIZ29L8F77M2A"], "BA", "20230301", "20230314", 21)
    
    def test_init(self):

        # Exception raised when end date is before start date
        with self.assertRaises(ValueError):
            AlphaVantageNewsDownloader(["API_KEY"], "BA", "20230314", "20230301", 21)
        with self.assertRaises(ValueError):
            AlphaVantageNewsDownloader(["API_KEY"], "BA", "20230301", "20230301", 21)
        
        # Exception raised when end date is in the future
        with self.assertRaises(ValueError):
            AlphaVantageNewsDownloader(["API_KEY"], "BA", "20230301", (datetime.now()+timedelta(days=1)).strftime("%Y%m%d"), 0)

    def test_download_raw_news_data(self):
        
        with self.assertRaises(ValueError):
            self.avnd.download_raw_news_data(date(2023, 3, 1), date(2023, 3, 14),2000)
        
        self.assertGreater(len(self.avnd.download_raw_news_data(date(2023, 3, 1), date(2023, 3, 14))), 1)

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
