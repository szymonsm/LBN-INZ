from app import db

class StockPrice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    adj_close = db.Column(db.Float)
    volume = db.Column(db.Integer)
    ticker = db.Column(db.String(10))

    def __init__(self, date, open, high, low, close, adj_close, volume, ticker):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adj_close = adj_close
        self.volume = volume
        self.ticker = ticker

# uuid,title,description,keywords,snippet,url,image_url,language,published_at,source,relevance_score,type,industry,match_score,sentiment_score

class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(1000))
    title = db.Column(db.String(1000))
    description = db.Column(db.String(1000))
    keywords = db.Column(db.String(1000))
    snippet = db.Column(db.String(1000))
    url = db.Column(db.String(1000))
    image_url = db.Column(db.String(1000))
    language = db.Column(db.String(1000))
    published_at = db.Column(db.DateTime)
    source = db.Column(db.String(1000))
    relevance_score = db.Column(db.Float)
    type = db.Column(db.String(1000))
    industry = db.Column(db.String(1000))
    match_score = db.Column(db.Float)
    sentiment_score = db.Column(db.Float)
    # summary = db.Column(db.String(1000))
    # overall_sentiment_score = db.Column(db.Float)
    # overall_sentiment_label = db.Column(db.String(10))
    # ticker_relevance_score = db.Column(db.Float)
    # ticker_sentiment_score = db.Column(db.Float)
    # ticker_sentiment_label = db.Column(db.String(10))
    # time_published = db.Column(db.DateTime)
    ticker = db.Column(db.String(10))

    def __init__(self, uuid,title,description,keywords,snippet,url,image_url,language,published_at,source,relevance_score,type,industry,match_score,sentiment_score,ticker):
        self.uuid = uuid
        self.title = title
        self.description = description
        self.keywords = keywords
        self.snippet = snippet
        self.url = url
        self.image_url = image_url
        self.language = language
        self.published_at = published_at
        self.source = source
        self.relevance_score = relevance_score
        self.type = type
        self.industry = industry
        self.match_score = match_score
        self.sentiment_score = sentiment_score
        # self.summary = summary
        # self.overall_sentiment_score = overall_sentiment_score
        # self.overall_sentiment_label = overall_sentiment_label
        # self.ticker_relevance_score = ticker_relevance_score
        # self.ticker_sentiment_score = ticker_sentiment_score
        # self.ticker_sentiment_label = ticker_sentiment_label
        # self.time_published = time_published
        self.ticker = ticker