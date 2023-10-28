import yaml
from flask import Flask, render_template, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect
from flask_migrate import Migrate
import pandas as pd
from datetime import datetime

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

app = Flask(__name__)
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'  # Change to your database URL
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

running_daemons = {}

@app.route('/')
def index():
    print('index')
    with open('config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    currency_options = config['options']['currency']
    time_period_options = config['options']['time_period']
    return render_template('index.html', currency_options=currency_options, time_period_options=time_period_options, running_daemons=running_daemons)

@app.route('/download_data/<type>/<ticker>/<interval>/<month>')
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
        pass
    else:
        raise ValueError(f'Unknown type: {type}')
    
    return redirect(url_for('index'))

@app.route('/display_tables')
def display_tables():
    stock_prices = StockPrice.query.all()
    columns = [column.key for column in inspect(StockPrice).c]
    return render_template('display_tables.html', stock_prices=stock_prices, columns=columns)

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