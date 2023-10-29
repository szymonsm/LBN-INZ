from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import tqdm
from CODE.language_models.sentiment_model import SentimentModel

class Vader(SentimentModel):
    """
    Class for Vader model that predicts sentiment of a text.
    """
    sid_obj = SentimentIntensityAnalyzer()
    def sentiment_scores(self, sentence):
        sentiment_dict = self.sid_obj.polarity_scores(sentence)
        return sentiment_dict

    def __init__(self, device: int = -1) -> None:
        self.name = "vader"
        self.pipe = self.sentiment_scores

    def pipeline_predict_sentiment(self, texts: str | list[str]) -> pd.DataFrame:
        """
        Predicts sentiment of a text or list of texts.

        :param texts: str or list[str], text(s) to predict sentiment of
        """
        print("Predicting...")
        predictions = []
        for text in tqdm.tqdm(texts):
            tmp_prediction = self.pipe(text)
            predictions.append(tmp_prediction)
        return pd.DataFrame(predictions)
    
    # def add_predictions_to_df(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Adds predictions to a dataframe.

    #     :param df: pd.DataFrame, dataframe to add predictions to
    #     :param predictions: list, predictions to add to dataframe
    #     :return: pd.DataFrame, dataframe with predictions added
    #     """
    #     return pd.concat([df, predictions], axis=1)
    

def main() -> None:
    # This is just usage example, not part of the class

    vader = Vader()

    df = pd.read_csv("DATA/alphavantage/news/BA/BA_20230315_20230430.csv")
    predictions = vader.pipeline_predict_sentiment(list(df["summary"]))
    df_new = Vader.add_predictions_to_df(df, predictions)
    print(df_new)
    
if __name__ == "__main__":
    main()
