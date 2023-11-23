from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tqdm
import pandas as pd

class BartLargeMNLI:
    """
    Class for BART model that predicts classes of a text. One can use own classes or use the default ones.
    """

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    def __init__(self):
        self.pipe = None

    def initialize_model(self, device: int = -1) -> None:
        """
        Initializes the model.

        :param device: int, -1 for CPU, 0 for GPU
        """
        self.pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    
    def predict_classes(self, texts: str | list[str], classes: list[str], multi_label: bool = True) -> list:
        """
        Predicts classes of a text or list of texts.

        :param texts: str or list[str], text(s) to predict classes of
        :param classes: list[str], classes to predict
        :param multi_label: bool, whether to predict multiple classes or not
        :return: list, predictions
        """
        predictions = []
        for text in tqdm.tqdm(texts):
            predictions.append(self.pipe(text, classes, multi_label=multi_label))
        return predictions
        # return self.pipe(texts, classes, multi_label=multi_label)
        # return predictions
    
    def add_predictions_to_df(df: pd.DataFrame, predictions: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
        """
        Adds predictions to a dataframe.

        :param df: pd.DataFrame, dataframe to add predictions to
        :param predictions: list, predictions to add to dataframe
        :return: pd.DataFrame, dataframe with predictions added
        """
        df_bart = pd.json_normalize(predictions)
        df_bart = pd.concat([df_bart.drop(['scores'], axis=1), df_bart['scores'].apply(pd.Series)], axis=1)
        df_bart.rename(columns=dict(zip(range(len(classes)),classes)), inplace=True)
        df_bart = df_bart.drop(['labels'], axis=1)
        return pd.concat([df, df_bart], axis=1)
    
def main():
    BartLargeMNLI.model.save_pretrained("MODELS/bart-large-mnli/")
    
if __name__ == "__main__":
    main()