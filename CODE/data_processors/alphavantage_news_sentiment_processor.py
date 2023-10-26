from CODE.language_models.finbert import FinBERT
import pandas as pd

class AVNewsSentimentProcessor:

    def __init__(self, device: int = -1):
        self.device = device # If you have GPU, set device=0, else set device=-1
        self.finbert = FinBERT(device)
        self.dfs = []

    def process_single_df(self, df: pd.Dataframe):
        predictions = self.finbert.pipeline_predict_sentiment(list(df["summary"]))
        df_new = FinBERT.add_predictions_to_df(df, predictions)
        return df_new
    


def main():

    # This is just usage example, not part of the class
    # If you have GPU, set device=0, else set device=-1
    device = -1

    avnsp = AVNewsSentimentProcessor(device)

    df = pd.read_csv("DATA/alphavantage/news/BA/BA_20230301_20230314.csv")
    df_new = avnsp.process_single_df(df)
    df_new.to_csv("DATA/finbert/alphavantage/BA/BA_20230301_20230314_finbert.csv")
    

if __name__ == "main":
    main()