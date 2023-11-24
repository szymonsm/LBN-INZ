from CODE.data_downloaders.alphavantage.alphavantage_news_downloader import AlphaVantageNewsDownloader
from CODE.data_downloaders.alphavantage.alphavantage_stock_price_downloader import AlphaVantageStockPriceDownloader
from unittest import TestCase
from datetime import date, timedelta, datetime

class AlphaVantageNewsDownloaderTest(TestCase):

    avnd = AlphaVantageNewsDownloader(["BC1SIZ29L8F77M2A"], "BA", "20230301", "20230314", 21)
    
    def test_init(self):
        # self.assertEqual(self.avnd.begin_date, date(2023, 3, 1))
        # self.assertEqual(self.avnd.end_date, date(2023, 3, 14))

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
        
        # Following tests should be
        self.assertGreater(len(self.avnd.download_raw_news_data(date(2023, 3, 1), date(2023, 3, 14))), 1)
        # self.assertEqual(len(self.avnd.download_raw_news_data(date(2023, 3, 1), date(2023, 2, 28))), 0)
    
    def test_convert_raw_news_data(self):
        pass


class AlphaVantageStockPriceDownloaderTest(TestCase):

    # This test requires internet access
    def test_download_daily_ticker_data(self):
        avspd = AlphaVantageStockPriceDownloader(["BC1SIZ29L8F77M2A"], "BA")
        data = avspd.download_daily_ticker_data()
        self.assertGreater(len(data), 0)

        avspd = AlphaVantageStockPriceDownloader(["WRONG API KEY"], "BA")
        data = avspd.download_daily_ticker_data()
        self.assertEqual(len(data), 0)

        avspd = AlphaVantageStockPriceDownloader(["BC1SIZ29L8F77M2A"], "NOT EXISTING TICKER")
        data = avspd.download_daily_ticker_data()
        self.assertEqual(len(data), 0)

    def test_download_intraday_ticker_data(self):
        
        avspd = AlphaVantageStockPriceDownloader(["BC1SIZ29L8F77M2A"], "BA")

        # Wrong interval
        with self.assertRaises(ValueError):
            avspd.download_intraday_ticker_data("2023-11", 23)

        data = avspd.download_intraday_ticker_data("2023-09", 60)

        self.assertNotEqual(data, None)

        
