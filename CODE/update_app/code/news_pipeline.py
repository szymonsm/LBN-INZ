import pandas as pd
import nltk
from CODE.update_app.code.language_models.news_models import BartLargeMNLI, FinBERT, Vader
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from functools import partial


class NewsSentimentProcessor:

    def __init__(self, ticker: str, df_news: pd.DataFrame, device: int = -1) -> None:
        self.ticker = ticker
        self.df = df_news
        self.device = device

    @staticmethod
    def trim_text_to_tokens(text, max_tokens=320):
        tokens = word_tokenize(text)
        trimmed_tokens = tokens[:max_tokens]
        trimmed_text = ' '.join(trimmed_tokens)
        return trimmed_text
    
    @staticmethod
    def save_to_dir(result_df: pd.DataFrame, save_path: str) -> None:
        result_df.to_csv(save_path, index=False)

    def convert_text(self) -> None:
        self.df["description"] = self.df["description"].fillna("")
        self.df["snippet"] = self.df["snippet"].fillna("")
        self.df["text"] = self.df['title']+". "+self.df["description"]+". "+self.df["snippet"]
        self.df["text_trim"] = self.df["text"].apply(self.trim_text_to_tokens)

    def process_finbert(self) -> None:
        finbert = FinBERT(self.device, False)
        predictions = finbert.pipeline_predict_sentiment(list(self.df["text_trim"]))
        self.df = FinBERT.add_predictions_to_df(self.df, predictions)

    def process_bart(self) -> None:
        bart_large_mnli = BartLargeMNLI()
        bart_large_mnli.initialize_model(device=self.device)

        for classes in [["bullish", "bearish"], ["past", "future"], ["influential", "redundant"], ["trustworthy", "untrustworthy"], ["clickbait", "not clickbait"]]:
            print(self.df['text'])
            predictions = bart_large_mnli.predict_classes(list(self.df['text']), classes, multi_label=False)
            self.df = BartLargeMNLI.add_predictions_to_df(self.df, predictions, classes)

    def process_vader(self) -> None:
        vader = Vader()
        predictions = vader.pipeline_predict_sentiment(list(self.df['text']))
        self.df = Vader.add_predictions_to_df(self.df, predictions)

    def add_columns(self) -> None:
        self.df["finbert_Score"] = self.df["finbert_positive"] - self.df["finbert_negative"]
        self.df["bart_Score"] = self.df["bullish"] - self.df["bearish"]
        self.df["vader_Score"] = self.df["compound"]

        for model in ["finbert", "bart", "vader"]:
            for class_ in ["future", "influential", "trustworthy", "clickbait"]:
                self.df[f"{class_}_{model}"] = self.df[f"{model}_Score"] * self.df[class_]

        self.df['published_at'] = pd.to_datetime(self.df['published_at'])
        self.df['date_8'] = self.df['published_at'] - pd.Timedelta(hours=8)

    def process_single_df(self) -> pd.DataFrame:
        self.df = self.df.loc[:,["date_8", "future", "influential", "trustworthy", "not clickbait", "finbert_Score", "bart_Score", "vader_Score", "future_finbert", "future_bart", "future_vader", "influential_finbert", "influential_bart", "influential_vader", "trustworthy_finbert", "trustworthy_bart", "trustworthy_vader", "clickbait_finbert", "clickbait_bart", "clickbait_vader"]]
        self.df['date_8'] = pd.to_datetime(self.df['date_8'], format="%Y-%m-%dT%H:%M:%S.000000Z")

        self.df['day'] = self.df['date_8'].dt.date
        result_df = self.df.groupby('day').mean().reset_index()
        return result_df
    

    def combine_with_existing(self, ex_df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
        result_df = pd.concat([ex_df, result_df], ignore_index=True)
        return result_df

    def launch(self):
        self.convert_text()
        self.process_finbert()
        self.process_bart()
        self.process_vader()
        self.add_columns()
        result_df = self.process_single_df()
        return result_df

def main():

    df_news = pd.read_csv("D:/pw/Thesis/LBN-INZ/DATA/final_news/TSLA_20210101_20231126.csv")

    news_sentiment_processor = NewsSentimentProcessor("TSLA", df_news, device=-1)
    news_sentiment_processor.launch()

if __name__ == "__main__":
    main()