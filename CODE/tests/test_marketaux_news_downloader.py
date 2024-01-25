from CODE.update_app.code.news_collector import MarketauxNewsDownloader
from unittest import TestCase
from datetime import date, timedelta, datetime

class MarketauxNewsDownloaderTest(TestCase):

    api_keys = ["SlbjFRNM8piSyVQsa1XaE6HzPQ2L1p1qMZPNsCHy"]
    ticker_name = "BA"
    begin_date = "2022-12-26"
    end_date = "2022-12-31"
    downloader = MarketauxNewsDownloader(api_keys, ticker_name, begin_date, end_date)
    
    def test_init(self):
        # self.assertEqual(self.avnd.begin_date, date(2023, 3, 1))
        # self.assertEqual(self.avnd.end_date, date(2023, 3, 14))

        # Exception raised when end date is before start date
        with self.assertRaises(ValueError):
            MarketauxNewsDownloader(self.api_keys, "BA", "2023-03-14", "2023-03-01")
        
        # Exception raised when end date is in the future
        with self.assertRaises(ValueError):
            MarketauxNewsDownloader(self.api_keys, "BA", "2023-03-01", (datetime.now()+timedelta(days=1)).strftime("%Y-%m-%d"))

    def test_download_raw_news_data(self):
        
        self.assertGreater(len(self.downloader.download_raw_news_data(date(2023, 3, 1), 1)), 0)
        # self.assertEqual(len(self.avnd.download_raw_news_data(date(2023, 3, 1), date(2023, 2, 28))), 0)
