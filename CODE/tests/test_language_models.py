from unittest import TestCase
from CODE.language_models.finbert import FinBERT
# from CODE.language_models.bart_large_mnli import BartLargeMNLI
from CODE.language_models.vader import Vader


class TestLanguageModels(TestCase):

    def test_finbert(self):
        finbert = FinBERT()
        prediction = finbert.pipeline_predict_sentiment(["Apple is gonna skyrocket by 20%"])
        print(prediction["finbert_positive"][0])

    # def test_bart_large_mnli(self):
    #     bart = BartLargeMNLI()
    #     bart.initialize_model()
    #     print(bart.predict_classes("Apple is gonna skyrocket by 20%", ["Bullish", "Bearish"]))

    def test_vader(self):
        vader = Vader()
        print(vader.sentiment_scores("Apple is gonna skyrocket by 20%"))
        print(vader.pipeline_predict_sentiment("Apple is gonna skyrocket by 20%"))
    