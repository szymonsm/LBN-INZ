import yaml
from flask import Flask, render_template, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect
from flask_migrate import Migrate
import pandas as pd
from datetime import datetime, timedelta
import requests
import plotly.graph_objs as go
import plotly.express as px
import re
import os

ALLOWED_INTERVALS = [1, 5, 15, 30, 60]

class AlphaVantageStockPriceDownloader:
    """
    Class for downloading stock price data from AlphaVantage. API key(s) are required.
    """
    def __init__(self, api_keys: list[str], ticker: str):
        """
        :param api_keys: list[str], list of API keys
        :param ticker: str, ticker to download news for (i.e. AAPL)
        """
        self.api_keys = api_keys
        self.ticker = ticker


    def download_daily_ticker_data(self) -> pd.DataFrame | None:
        """
        Downloads historical daily ticker data for 20+ years from AlphaVantage.

        :return: pd.DataFrame, daily ticker data or None if no data was downloaded
        """

        for api_key in self.api_keys:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.ticker}&outputsize=full&apikey={api_key}&datatype=csv'
            df_daily = pd.read_csv(url)
            print(f"Number of downloaded rows: {len(df_daily)}")
            if len(df_daily)>2:
                return df_daily
        return None 
    
    def download_intraday_ticker_data(self, month: str, interval: int) -> pd.DataFrame | None:
        """
        Downloads historical intraday ticker data for a given month and time interval from AlphaVantage.

        :param month: str, month in format YYYY-MM
        :param interval: int, interval in minutes (1, 5, 15, 30, 60)
        :return: pd.DataFrame, intraday ticker data or None if no data was downloaded
        """
        if interval not in ALLOWED_INTERVALS:
            raise ValueError(f"Given interval is not available, choose from this list: {ALLOWED_INTERVALS}")
        for api_key in self.api_keys:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.ticker}&interval={interval}min&month={month}&outputsize=full&apikey={api_key}&datatype=csv'
            df_daily = pd.read_csv(url)
            print(f"Number of downloaded rows: {len(df_daily)}")
            if len(df_daily)>2: 
                return df_daily
        return None 
    
class AlphaVantageNewsDownloader:
    """
    Class for downloading news data from AlphaVantage. API key(s) are required.
    """

    def __init__(self, api_keys: list[str], ticker: str, begin_date: str, end_date: str, days_per_request: int = 30) -> None:
        """
        :param api_keys: list[str], list of API keys
        :param ticker: str, ticker to download news for (i.e. AAPL)
        :param begin_date: str, begin date in format YYYYMMDD
        :param end_date: str, end date in format
        :param days_per_request: int, number of days per request (optimal is 30)
        """
        self.api_keys = api_keys
        self.ticker = ticker
        self.begin_date = datetime.strptime(begin_date, "%Y%m%d")
        self.end_date = datetime.strptime(end_date, "%Y%m%d")
        if self.end_date <= self.begin_date:
            raise ValueError("End date must be greater than begin date")
        if self.end_date > datetime.now():
            raise ValueError("End date must be less or equal than current date")
        self.days_per_request = days_per_request


    def download_raw_news_data(self, begin_date: datetime.date, end_date: datetime.date, limit: int = 1000) -> dict:
        """
        Downloads raw news data from AlphaVantage.

        :param begin_date: datetime.date, begin date
        :param end_date: datetime.date, end date
        :param limit: int, limit of news per request (max is 1000)
        :return: dict, raw news data
        """

        assert limit<=1000, "AlphaVantage supports news limit up to 1000 per request"

        begin_date_f = datetime.strftime(begin_date, "%Y%m%d")
        end_date_f = datetime.strftime(end_date, "%Y%m%d")

        for api_key in self.api_keys:
            print(f"Downloading raw news from {begin_date_f} to {end_date_f}...")
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&time_from={begin_date_f}T0000&time_to={end_date_f}T0000&limit={limit}&apikey={api_key}&datatype=csv"
            r = requests.get(url)
        
            data_news = r.json()
            if len(data_news)>1:
                print("Raw news downloaded correctly")
                return data_news
            print("WARNING: Raw news not downloaded correctly, you might have exceeded API limit")
        return {}


    def convert_raw_news_data(self, data_news: dict):
        """
        Converts raw news data to a dictionary that can be easily converted to a dataframe.

        :param data_news: dict, raw news data
        :return: dict, converted news data
        """

        dict_news = {"title":[], "url":[], "summary":[], "source": [], "topics": [], 
                     "category_within_source": [], "authors": [], "overall_sentiment_score":[],
                     "overall_sentiment_label":[], "ticker_relevance_score":[],
                     "ticker_sentiment_score":[], "ticker_sentiment_label":[],
                     "time_published":[]}
        
        ticker_sentiment_keys = ["ticker_relevance_score", "ticker_sentiment_score", "ticker_sentiment_label"]

        for event in data_news.get("feed", []):
            for key in dict_news:
                if key not in ticker_sentiment_keys:
                    dict_news[key].append(event.get(key, None))

            for ticker in event["ticker_sentiment"]:
                if ticker["ticker"]==self.ticker:
                    for key in ticker_sentiment_keys:
                        dict_news[key].append(ticker.get(key, None))
        print(f"Number of added news: {len(data_news.get('feed', []))}")
        return dict_news
    
    def download_multiple_data(self):
        """
        Downloads data for multiple dates from AlphaVantage.

        :return: dict, converted news data
        """
        dict_news = {"title":[], "url":[], "summary":[], "overall_sentiment_score":[],
                "overall_sentiment_label":[], "ticker_relevance_score":[],
                "ticker_sentiment_score":[], "ticker_sentiment_label":[],
                "time_published":[]}
        current_date = self.begin_date

        for _ in range(5):

            # How do I want it to work?
            # One cannot download data when end_date >= current_date, in that case
            # I want it to download data from current_date to end_date and return
            if current_date >= self.end_date:
                print("End date reached")
                return dict_news
            
            cur_end_date = current_date + timedelta(days=self.days_per_request)

            if cur_end_date >= self.end_date:
                cur_end_date = self.end_date
            
            data_news = self.download_raw_news_data(current_date, cur_end_date)
            if data_news=={}:
                self.end_date = current_date
                return dict_news
            dict_news_tmp = self.convert_raw_news_data(data_news)
            for key in dict_news:
                dict_news[key] += dict_news_tmp[key]
            
            current_date = cur_end_date
        self.end_date = current_date
        return dict_news

app = Flask(__name__)
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sda_db.db'  # Change to your database URL
db = SQLAlchemy(app)
migrate = Migrate(app, db)

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
        
running_daemons = {}

@app.route('/')
def index():
    print('index')
    with open('config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    currency_options = config['options']['currency']
    time_period_options = config['options']['time_period']
    return render_template('index.html', currency_options=currency_options, time_period_options=time_period_options, running_daemons=running_daemons)

@app.route('/download_data/<type>/<ticker>/<month>/<interval>')
def download_data(type, ticker, interval, month):
    # Get user input (e.g., ticker, month, interval)

    with open('config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    interval = int(interval)

    api_keys = config['api_keys']['alphavantage']

    if type == 'stock_price':
        # Use your data downloader class to fetch data (replace with actual code)
        downloader = AlphaVantageStockPriceDownloader(api_keys, ticker)
        df = downloader.download_intraday_ticker_data(month, interval)
        date_format = '%Y-%m-%d %H:%M:%S'

        # Store the data in the database
        for _, row in df.iterrows():
            stock_price = StockPrice(
                timestamp=datetime.strptime(row['timestamp'], date_format),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                ticker=ticker,
                interval=interval,
                month=month
            )
            db.session.add(stock_price)

        db.session.commit()
    elif type == 'news':
        downloader = AlphaVantageNewsDownloader(api_keys, ticker, month[:8], month[8:])
        df = pd.DataFrame.from_dict(downloader.download_multiple_data())
        date_format = '%Y%m%dT%H%M%S'

        for _, row in df.iterrows():
            news = News(
                title=row['title'],
                url=row['url'],
                summary=row['summary'],
                overall_sentiment_score=row['overall_sentiment_score'],
                overall_sentiment_label=row['overall_sentiment_label'],
                ticker_relevance_score=row['ticker_relevance_score'],
                ticker_sentiment_score=row['ticker_sentiment_score'],
                ticker_sentiment_label=row['ticker_sentiment_label'],
                time_published=datetime.strptime(row['time_published'], date_format),
                ticker=ticker            )
            db.session.add(news)
        db.session.commit()
    else:
        raise ValueError(f'Unknown type: {type}')
    
    return redirect(url_for('index'))

@app.route('/display_tables')
def display_tables():
    stock_prices = StockPrice.query.all()
    columns_prices = [column.key for column in inspect(StockPrice).c]
    news = News.query.all()
    columns_news = [column.key for column in inspect(News).c]
    return render_template('display_tables.html', stock_prices=stock_prices, news=news, columns_prices=columns_prices, columns_news=columns_news)

@app.route('/update_plot', methods=['POST'])
def update_plot():
    selected_currency = request.json.get('currency')
    selected_time_period = request.json.get('time_period')
    selected_data_type = request.json.get('data_type')

    # Modify the query to filter data based on selected criteria
    data = StockPrice.query.filter_by(
        ticker=selected_currency,
        interval=selected_time_period
    ).with_entities(StockPrice.timestamp, getattr(StockPrice, selected_data_type)).all()

    timestamps = [pd.to_datetime(row[0]) for row in data]
    price_data = [row[1] for row in data]
    df = pd.DataFrame(dict(
        x = timestamps,
        y = price_data
    ))
    print(df.head())

    # Create the Plotly figure
    # fig = go.Figure(data=go.Scatter(x=timestamps, y=price_data, mode='lines', name=selected_data_type))
    fig = px.line(df, x='x', y='y', title=f'{selected_currency} {selected_data_type} {selected_time_period}')

    # Convert the Plotly figure to JSON format for AJAX response
    plot_data = fig.to_json()
    return jsonify(plot_data)

# @app.route('/start_daemon/<currency>/<time_period>/<model>')
# def start_daemon_route(currency, time_period, model):
#     key = f'{currency}_{time_period}_{model}'
#     start_daemon(currency, time_period, model)
#     running_daemons[key] = f'Daemon for {currency}, {time_period}, {model} running...'
#     return 'Daemon started.'

# # Flask route to handle stopping a daemon
# @app.route('/stop_daemon/<key>')
# def stop_daemon_route(key):
#     stop_daemon(key)
#     return 'Daemon stopped.'