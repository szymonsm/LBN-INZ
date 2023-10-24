from app.webapp import db
# from .webapp import db


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
        