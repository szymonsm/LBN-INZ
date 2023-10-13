from transformers import pipeline
import pandas as pd
import tqdm


class FinBERT:
    """
    Class for FinBERT model that predicts sentiment of a text.
    """
    def __init__(self) -> None:
        self.pipe = None

    def initialize_model(self, device: int = -1) -> None:
        """
        Initializes the model.
        
        :param device: int, -1 for CPU, 0 for GPU
        """
        print("Initializing model...")
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device)
        print("Model initialized")

    def predict_sentiment(self, texts: str | list[str]) -> list:
        """
        Predicts sentiment of a text or list of texts.
        # TODO: add option to return probabilities

        :param texts: str or list[str], text(s) to predict sentiment of
        """
        print("Predicting...")
        # return tqdm(self.pipe(texts))
        predictions = []
        for text in tqdm.tqdm(texts):
            predictions.append(self.pipe(text))
        return predictions
    
    def add_predictions_to_df(self, df: pd.DataFrame, predictions: list) -> pd.DataFrame:
        """
        Adds predictions to a dataframe.

        :param df: pd.DataFrame, dataframe to add predictions to
        :param predictions: list, predictions to add to dataframe
        :return: pd.DataFrame, dataframe with predictions added
        """
        return pd.concat([df,pd.DataFrame(predictions)], axis=1)
    

def main() -> None:
    # This is just usage example, not part of the class
    # If you have GPU, set device=0, else set device=-1
    device = -1

    finbert = FinBERT()
    finbert.initialize_model(device=device)

    df = pd.read_csv("DATA/alphavantage/news/CRYPTOBTC/CRYPTOBTC_20230101_20231001_example.csv")
    predictions = finbert.predict_sentiment(list(df["summary"]))
    df_new = finbert.add_predictions_to_df(df, predictions)
    df_new.to_csv("test.csv")
    
if __name__ == "__main__":
    main()