import yaml
from flask import render_template, render_template, request, redirect, url_for
from app import app
from app.webapp import db
from app.code.notifications.daemon_manager import start_daemon, stop_daemon
from app.models import StockPrice  # Import the StockPrice model
from app.code.data_downloaders.alphavantage.alphavantage_stock_price_downloader import AlphaVantageStockPriceDownloader 
# from .webapp import db
# from .code.notifications.daemon_manager import start_daemon, stop_daemon
# from .models import StockPrice  # Import the StockPrice model
# from .code.data_downloaders.alphavantage.alphavantage_stock_price_downloader import AlphaVantageStockPriceDownloader 

# Store running daemons
running_daemons = {}

@app.route('/')
def index():
    print('index')
    with open('app/config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    currency_options = config['options']['currency']
    time_period_options = config['options']['time_period']
    return render_template('index.html', currency_options=currency_options, time_period_options=time_period_options, running_daemons=running_daemons)

@app.route('/download_data/<type>/<ticker>/<interval>/<month>')
def download_data(type, ticker, interval, month):
    # Get user input (e.g., ticker, month, interval)

    with open('app/config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    interval = int(interval)

    api_keys = config['api_keys']['alphavantage']

    if type == 'stock_price':
        # Use your data downloader class to fetch data (replace with actual code)
        downloader = AlphaVantageStockPriceDownloader(api_keys, ticker)
        df = downloader.download_intraday_ticker_data(month, interval)

        # Store the data in the database
        for _, row in df.iterrows():
            stock_price = StockPrice(
                timestamp=row['timestamp'],
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


@app.route('/start_daemon/<currency>/<time_period>/<model>')
def start_daemon_route(currency, time_period, model):
    key = f'{currency}_{time_period}_{model}'
    start_daemon(currency, time_period, model)
    running_daemons[key] = f'Daemon for {currency}, {time_period}, {model} running...'
    return 'Daemon started.'

# Flask route to handle stopping a daemon
@app.route('/stop_daemon/<key>')
def stop_daemon_route(key):
    stop_daemon(key)
    return 'Daemon stopped.'
