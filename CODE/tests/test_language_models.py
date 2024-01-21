from unittest import TestCase
from CODE.language_models.finbert import FinBERT
from CODE.language_models.bart_large_mnli import BartLargeMNLI
from CODE.language_models.vader import Vader


class TestLanguageModels(TestCase):

    def test_finbert(self):
        finbert = FinBERT()

        # Positive Test Case
        prediction = finbert.pipeline_predict_sentiment(["Apple is gonna skyrocket by 20%"])
        self.assertGreaterEqual(prediction["finbert_positive"][0], prediction["finbert_negative"][0])

        # Negative Test Case
        prediction = finbert.pipeline_predict_sentiment(["Apple stock is going to drop"])
        self.assertGreaterEqual(prediction["finbert_negative"][0], prediction["finbert_positive"][0])

        # Neutral Test Case
        prediction = finbert.pipeline_predict_sentiment(["Apples are red."])
        self.assertGreaterEqual(prediction["finbert_neutral"][0], max(prediction["finbert_positive"][0], prediction["finbert_negative"][0]))

    def test_bart(self):
        bart = BartLargeMNLI()
        bart.initialize_model()
        # Positive Test Case
        prediction = bart.predict_classes(["Apple is gonna skyrocket by 20%"], ["bullish", "bearish"])
        self.assertEqual(prediction[0]["labels"][0], "bullish")

        # Negative Test Case
        prediction = bart.predict_classes(["Apple stock is going to drop"], ["bullish", "bearish"])
        self.assertEqual(prediction[0]["labels"][0], "bearish")

    def test_vader(self):
        vader = Vader()
        
        # Positive Test Case
        scores = vader.sentiment_scores(["Apple is gonna skyrocket by 20%"])
        self.assertGreaterEqual(scores["pos"], scores["neg"])
        
        # Negative Test Case
        scores = vader.sentiment_scores(["Apple stock is going to drop"])
        self.assertGreaterEqual(scores["neg"], scores["pos"])

        # Neutral Test Case
        scores = vader.sentiment_scores(["Apples are red."])
        self.assertGreaterEqual(scores["neu"], max(scores["pos"], scores["neg"]))
    