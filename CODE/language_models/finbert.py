# Load model directly
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments
import pandas as pd
import tqdm
from CODE.language_models.sentiment_model import SentimentModel
import torch
import scipy


class FinBERT(SentimentModel):
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

    def __init__(self, device: int = -1, return_one_label: bool = False) -> None:
        self.name = "finbert"
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device) if return_one_label else self.predict_return_all_labels

    def pipeline_predict_sentiment(self, texts: str | list[str]) -> pd.DataFrame:
        """
        Predicts sentiment of a text or list of texts.

        :param texts: str or list[str], text(s) to predict sentiment of
        """
        print("Predicting...")

        if isinstance(texts, str):
            texts = [texts]
        # return tqdm(self.pipe(texts))
        predictions = []
        for text in tqdm.tqdm(texts):
            tmp_prediction = self.pipe(text)
            # predictions += tmp_prediction - WORKS WHEN ONE LABEL IS RETURNED
            predictions.append(tmp_prediction)
        predictions = pd.json_normalize(predictions)
        predictions.rename(columns={'positive': 'finbert_positive', 'neutral': 'finbert_neutral', 'negative': 'finbert_negative'}, inplace=True)
        return predictions
    

def main() -> None:
    # This is just usage example, not part of the class
    # If you have GPU, set device=0, else set device=-1
    device = -1

    finbert = FinBERT(device)

    df = pd.read_csv("DATA/alphavantage/news/BA/BA_20230315_20230430.csv")
    # predictions = finbert.pipeline_predict_sentiment(list(df["summary"]))
    predictions = finbert.pipeline_predict_sentiment(["Some text"])
    # df_new = FinBERT.add_predictions_to_df(df, predictions)
    # df_new.to_csv("test.csv")

    # FinBERT.model.save_pretrained("MODELS/finbert/")
    
if __name__ == "__main__":
    main()
