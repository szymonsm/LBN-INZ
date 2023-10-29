# Load model directly
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments
import pandas as pd
import tqdm
import os
import re
import datetime


class FinBERT:
    """
    Class for FinBERT model that predicts sentiment of a text.
    """

    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    def __init__(self, device: int = -1) -> None:
        self.name = "finbert"
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device)

    def pipeline_predict_sentiment(self, texts: str | list[str]) -> pd.DataFrame:
        """
        Predicts sentiment of a text or list of texts.
        # TODO: add option to return probabilities

        :param texts: str or list[str], text(s) to predict sentiment of
        """
        print("Predicting...")
        # return tqdm(self.pipe(texts))
        predictions = []
        for text in tqdm.tqdm(texts):
            tmp_prediction = self.pipe(text)
            predictions += tmp_prediction

        predictions = pd.json_normalize(predictions)
        predictions.rename(columns={"label": "finbert_label", "score": "finbert_score"}, inplace=True)
        return predictions
    
    def add_predictions_to_df(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Adds predictions to a dataframe.

        :param df: pd.DataFrame, dataframe to add predictions to
        :param predictions: list, predictions to add to dataframe
        :return: pd.DataFrame, dataframe with predictions added
        """
        return pd.concat([df, predictions], axis=1)
    

def main() -> None:
    # This is just usage example, not part of the class
    # If you have GPU, set device=0, else set device=-1
    device = -1

    finbert = FinBERT(device)

    df = pd.read_csv("DATA/alphavantage/news/BA/BA_20230315_20230430.csv")
    predictions = finbert.pipeline_predict_sentiment(list(df["summary"]))

    df_new = FinBERT.add_predictions_to_df(df, predictions)
    # df_new.to_csv("test.csv")

    # FinBERT.model.save_pretrained("MODELS/finbert/")
    
if __name__ == "__main__":
    main()
