from CODE.data_downloaders.alphavantage.alphavantage_news_downloader import AlphaVantageNewsDownloader
from CODE.data_downloaders.alphavantage.alphavantage_stock_price_downloader import AlphaVantageStockPriceDownloader
import datetime
from CODE.data_processors.news_sentiment_processor import NewsSentimentProcessor
from CODE.language_models.bart_large_mnli import BartLargeMNLI

from CODE.language_models.finbert import FinBERT
from CODE.language_models.text_embedder import TextEmbedder

def main():
    
    day = "2023-11-04"
    ticker = "BA"
    
    # Initialize Downloaders
    alphavantage_api_key = "BC1SIZ29L8F77M2A"
    news_downloader = AlphaVantageNewsDownloader([alphavantage_api_key], ticker, "20231104", "20231105", 2)
    price_downloader = AlphaVantageStockPriceDownloader([alphavantage_api_key], ticker)

    # Download Data
    dict_news = news_downloader.download_raw_news_data(datetime.datetime.strptime(day, "%Y-%m-%d"), datetime.datetime.strptime(day, "%Y-%m-%d"))
    df_news = news_downloader.convert_raw_news_data(dict_news)

    df_price_daily = price_downloader.download_daily_ticker_data()
    df_price_intraday = price_downloader.download_intraday_ticker_data(day[:7], 15)

    # Save Data
    news_downloader.save_to_dir(dict_news)
    price_downloader.save_to_dir(df_price_daily, "daily")
    price_downloader.save_to_dir(df_price_intraday, "intraday", day[:7], 15)

    # Process Data
    finbert_sentiment_processor = NewsSentimentProcessor("finbert")
    vader_sentiment_processor = NewsSentimentProcessor("vader")

    ## FinBERT
    df_news = finbert_sentiment_processor.process_single_df(df_news)

    ## Vader
    df_news = vader_sentiment_processor.process_single_df(df_news)

    ## BartLargeMNLI
    bart_large_mnli = BartLargeMNLI()
    bart_large_mnli.initialize_model()
    predictions = bart_large_mnli.predict_classes(list(df_news["summary"]), ["bullish", "bearish", "neutral"], multi_label=False)
    df_news = BartLargeMNLI.add_predictions_to_df(df_news, predictions)

    ## Embeddings
    te = TextEmbedder('BAAI/bge-base-en-v1.5', 'BAAI/bge-base-en-v1.5', -1)
    embeddings = te.encode(list(df_news["summary"]))
    df_news = TextEmbedder.add_predictions_to_df(df_news, embeddings)

    
    df_news.to_csv("full_pipeline_test.csv")


if __name__ == "__main__":
    main()