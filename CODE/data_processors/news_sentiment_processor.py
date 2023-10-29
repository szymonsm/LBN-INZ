from CODE.language_models.finbert import FinBERT
from CODE.language_models.vader import Vader
import pandas as pd
import os
import re
import datetime
from CODE.language_models.sentiment_model import SentimentModel

class NewsSentimentProcessor:

    NAME_MODEL = {"vader": Vader, "finbert": FinBERT}

    def __init__(self, model_name: str, device: int | None = None):
        print("Loading sentiment model...")
        self.device = device # If you have GPU, set device=0, else set device=-1
        self.sentiment_model: SentimentModel = self.NAME_MODEL[model_name](device) if device is not None else self.NAME_MODEL[model_name]()
        print(f"{self.sentiment_model.name} loaded.")

    def load_dfs(self, dir_path: str) -> list[pd.DataFrame]:
        """
        Loads dataframes from a directory.

        :param dir_path: str, directory path
        """
        dfs = []
        for file in os.listdir(dir_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(dir_path, file))
                dfs.append(df)
        return dfs

    def process_single_df(self, df: pd.DataFrame):
        predictions = self.sentiment_model.pipeline_predict_sentiment(list(df["summary"]))
        df_new = SentimentModel.add_predictions_to_df(df, predictions)
        return df_new
    
    def process_multiple_dfs(self, dfs: list[pd.DataFrame]):
        """
        Processes multiple dataframes and concatenates them into one dataframe

        :param dfs: list[pd.DataFrame], list of dataframes to process
        """
        dfs_new = []
        for df in dfs:
            predictions = self.sentiment_model.pipeline_predict_sentiment(list(df["summary"]))
            df_new = SentimentModel.add_predictions_to_df(df, predictions)
            dfs_new.append(df_new)
        return pd.concat(dfs_new, axis=0)

    def save_to_dir(self, df: pd.DataFrame, ticker: str, begin_date: datetime.date, end_date: datetime.date, data_provider: str = "alphavantage") -> None:
        """
        Saves news data to a directory.

        :param dict_news: dict, news data
        """
        print("Saving...")
        news_path = os.path.join("DATA", self.sentiment_model.name, data_provider)
        dir_path = os.path.join(news_path, re.sub(r'[^A-Za-z0-9]+', '', ticker))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = os.path.join(dir_path, f"{re.sub(r'[^A-Za-z0-9]+', '', ticker)}_{datetime.datetime.strftime(begin_date, '%Y%m%d')}_{datetime.datetime.strftime(end_date, '%Y%m%d')}_{self.sentiment_model.name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved file to path: {path}")

    def process_dir(self, dir_path: str) -> pd.DataFrame:
        """
        Processes directory of dataframes.

        :param dir_path: str, directory path
        :return: pd.DataFrame, conacatenated dataframe with finbert predictions
        """
        dfs = self.load_dfs(dir_path)
        df_new = self.process_multiple_dfs(dfs)
        return df_new
    

    
def main():
    # This is just usage example, not part of the class
    # If you have GPU, set device=0, else set device=-1
    device = -1
    avnsp = NewsSentimentProcessor("vader", device)
    print("Loading dataframe...")
    # df = pd.read_csv("DATA/alphavantage/news/BA/BA_20230301_20230314.csv")
    # df_new = avnsp.process_single_df(df)
    # avnsp.save_to_dir(df_new, "BA", datetime.date(2023,3,1), datetime.date(2023,3,14))
    df_new = avnsp.process_dir(os.path.join("DATA","alphavantage", "news", "BA"))
    avnsp.save_to_dir(df_new, "BA", datetime.date(2023,3,1), datetime.date(2023,4,30))
    

if __name__ == "__main__":
    main()