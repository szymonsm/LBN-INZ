from CODE.data_downloaders.alphavantage.alphavantage_news_downloader import AlphaVantageNewsDownloader
from CODE.data_downloaders.alphavantage.alphavantage_stock_price_downloader import AlphaVantageStockPriceDownloader
import datetime
import pandas as pd
from CODE.data_processors.news_sentiment_processor import NewsSentimentProcessor
from CODE.language_models.bart_large_mnli import BartLargeMNLI
from CODE.language_models.finbert import FinBERT
from CODE.language_models.text_embedder import TextEmbedder



def add_sentiment_final_scores(df_new: pd.DataFrame, classes_list: list[list[str]]) -> pd.DataFrame:

    df_new["finbert_Score"] = df_new["finbert_positive"] - df_new["finbert_negative"]
    df_new["bart_Score"] = df_new["bullish"] - df_new["bearish"]
    df_new["vader_Score"] = df_new["compound"]

    # future, past
    for classes in classes_list:
        for model in ["finbert", "bart", "vader"]:
            df_new[f"{classes}_{model}"] = df_new[f"{model}_Score"] * df_new[classes]
    return df_new

def group_data_with_scores(df_all_cols: pd.DataFrame) -> pd.DataFrame:
    df_all_cols['time_published'] = pd.to_datetime(df_all_cols['time_published'], format="%Y%m%dT%H%M%S")

    # Extract day from 'time_published'
    df_all_cols['day'] = df_all_cols['time_published'].dt.date

    # Group by 'day' and calculate mean for other columns
    result_df = df_all_cols.groupby('day').mean().reset_index()
    result_df


def main():
    
    # day = "2023-11-04"
    # ticker = "BA"
    
    # # Initialize Downloaders
    # alphavantage_api_key = "BC1SIZ29L8F77M2A"
    # news_downloader = AlphaVantageNewsDownloader([alphavantage_api_key], ticker, "20231104", "20231105", 2)
    # price_downloader = AlphaVantageStockPriceDownloader([alphavantage_api_key], ticker)

    # # Download Data
    # dict_news = news_downloader.download_raw_news_data(datetime.datetime.strptime(day, "%Y-%m-%d"), datetime.datetime.strptime(day, "%Y-%m-%d"))
    # df_news = news_downloader.convert_raw_news_data(dict_news)

    # df_price_daily = price_downloader.download_daily_ticker_data()
    # df_price_intraday = price_downloader.download_intraday_ticker_data(day[:7], 15)

    # # Save Data
    # news_downloader.save_to_dir(dict_news)
    # price_downloader.save_to_dir(df_price_daily, "daily")
    # price_downloader.save_to_dir(df_price_intraday, "intraday", day[:7], 15)

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
    for classes in [["bullish", "bearish"], ["past", "future"], ["influential", "redundant"], ["trustworthy", "untrustworthy"], ["clickbait", "not clickbait"]]:
        predictions = bart_large_mnli.predict_classes(list(df_news['title']+". "+df_news["summary"]), classes, multi_label=False)
        df_news = BartLargeMNLI.add_predictions_to_df(df_news, predictions, classes)

    classes_list = ["future", "influential", "trustworthy", "not clickbait"]
    df_new = add_sentiment_final_scores(df_news, classes_list)
    df_all_cols = df_new.loc[:,["time_published", "future", "influential", "trustworthy", "not clickbait", "finbert_Score", "bart_Score", "vader_Score", "future_finbert", "future_bart", "future_vader", "influential_finbert", "influential_bart", "influential_vader", "trustworthy_finbert", "trustworthy_bart", "trustworthy_vader", "clickbait_finbert", "clickbait_bart", "clickbait_vader"]]
    df_all_cols = group_data_with_scores(df_all_cols)
    
    # ## Embeddings
    # te = TextEmbedder('BAAI/bge-base-en-v1.5', 'BAAI/bge-base-en-v1.5', -1)
    # embeddings = te.encode(list(df_news["summary"]))
    # df_news = TextEmbedder.add_predictions_to_df(df_news, embeddings)

    
    df_all_cols.to_csv("full_pipeline_test.csv")


if __name__ == "__main__":
    main()