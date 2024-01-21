# Load model directly
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments
import pandas as pd
import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import os
import scipy


# FinBERT
class FinBERT:
    """
    Class for FinBERT model that predicts sentiment of a text.
    """

    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    def predict_return_all_labels(self, text: str):
      inputs = self.tokenizer(text, return_tensors="pt")
      with torch.no_grad():
        logits = self.model(**inputs).logits
        scores = {k: v for k, v in zip(self.model.config.id2label.values(), scipy.special.softmax(logits.numpy().squeeze()))}
      return scores

    def __init__(self, device: int = -1, return_one_label: bool = True) -> None:
        self.name = "finbert"
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device) if return_one_label else self.predict_return_all_labels

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
            # predictions += tmp_prediction - WORKS WHEN ONE LABEL IS RETURNED
            predictions.append(tmp_prediction)
        predictions = pd.json_normalize(predictions)
        predictions.rename(columns={'positive': 'finbert_positive', 'neutral': 'finbert_neutral', 'negative': 'finbert_negative'}, inplace=True)
        return predictions

    def add_predictions_to_df(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Adds predictions to a dataframe.

        :param df: pd.DataFrame, dataframe to add predictions to
        :param predictions: list, predictions to add to dataframe
        :return: pd.DataFrame, dataframe with predictions added
        """
        return pd.concat([df, predictions], axis=1)



# BartLargeMNLI
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
        self.pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", tokenizer="facebook/bart-large-mnli", device=device)

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

# VADER
class Vader:
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

    def add_predictions_to_df(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Adds predictions to a dataframe.

        :param df: pd.DataFrame, dataframe to add predictions to
        :param predictions: list, predictions to add to dataframe
        :return: pd.DataFrame, dataframe with predictions added
        """
        return pd.concat([df, predictions], axis=1)


# Text Embedder
class TextEmbedder:

    def __init__(self, model_name: str, tokenizer_name: str, device: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, device=device)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts: str | list[str], max_length: int = 128) -> torch.Tensor:
        """
        Encodes text(s) into embeddings.

        :param texts: str or list[str], text(s) to encode
        :param max_length: int, max length of text
        :return: torch.Tensor, embeddings
        """
        self.model.eval()
        if isinstance(texts, str):
            texts = [texts]
        print("Started tokenizing...")
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        print("Tokenizing done, now computing embeddings...")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        print("Created embeddings")
        return embeddings

    def add_predictions_to_df(df: pd.DataFrame, predictions: torch.tensor) -> pd.DataFrame:
        """
        Adds predictions to a dataframe.

        :param df: pd.DataFrame, dataframe to add predictions to
        :param predictions: torch.tensor, embeddings to add to dataframe
        :return: pd.DataFrame, dataframe with predictions added
        """
        df["embeddings"] = predictions.tolist()
        return df

