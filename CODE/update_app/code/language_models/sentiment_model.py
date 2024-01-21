from abc import ABC, abstractmethod
import pandas as pd


class SentimentModel:

    def __init__(self):
        self.pipe = None

    @abstractmethod
    def pipeline_predict_sentiment(self, texts: str | list[str]) -> pd.DataFrame:
        """
        Predicts sentiment of a text or list of texts.

        :param texts: str or list[str], text(s) to predict sentiment of
        """
        pass
    
    def add_predictions_to_df(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Adds predictions to a dataframe.

        :param df: pd.DataFrame, dataframe to add predictions to
        :param predictions: list, predictions to add to dataframe
        :return: pd.DataFrame, dataframe with predictions added
        """
        return pd.concat([df, predictions], axis=1)