from app import db

class StockPrice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Integer)
    ticker = db.Column(db.String(10))
    interval = db.Column(db.Integer)
    month = db.Column(db.String(10))

    def __init__(self, timestamp, open, high, low, close, volume, ticker, interval, month):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.ticker = ticker
        self.interval = interval
        self.month = month

class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(1000))
    url = db.Column(db.String(1000))
    summary = db.Column(db.String(1000))
    overall_sentiment_score = db.Column(db.Float)
    overall_sentiment_label = db.Column(db.String(10))
    ticker_relevance_score = db.Column(db.Float)
    ticker_sentiment_score = db.Column(db.Float)
    ticker_sentiment_label = db.Column(db.String(10))
    time_published = db.Column(db.DateTime)
    ticker = db.Column(db.String(10))

    def __init__(self, title, url, summary, overall_sentiment_score, overall_sentiment_label, ticker_relevance_score, ticker_sentiment_score, ticker_sentiment_label, time_published, ticker):
        self.title = title
        self.url = url
        self.summary = summary
        self.overall_sentiment_score = overall_sentiment_score
        self.overall_sentiment_label = overall_sentiment_label
        self.ticker_relevance_score = ticker_relevance_score
        self.ticker_sentiment_score = ticker_sentiment_score
        self.ticker_sentiment_label = ticker_sentiment_label
        self.time_published = time_published
        self.ticker = ticker