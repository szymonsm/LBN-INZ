from app import app, db
from flask import render_template, request, redirect, url_for, jsonify
from models import StockPrice, News
from sqlalchemy import inspect, desc
from datetime import datetime
import yaml
import json
import pandas as pd
from code.alphavantage_stock_price_downloader import AlphaVantageStockPriceDownloader
from code.alphavantage_news_downloader import AlphaVantageNewsDownloader

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

@app.route('/update_plots', methods=['POST'])
def update_plot():
    selected_currency = request.json.get('currency')
    selected_time_period = request.json.get('time_period')
    selected_data_type = request.json.get('data_type')

    # Modify the query to filter data based on selected criteria
    # As for now I am sending all the data to the frontend
    data = StockPrice.query.filter_by(
        ticker=selected_currency,
        interval=selected_time_period
    ).with_entities(StockPrice.timestamp, getattr(StockPrice, selected_data_type)).all()

    # Need of sending timestamps as strings to the frontend
    timestamps = [str(row[0]) for row in data]
    price_data = [row[1] for row in data]
    fig_json = json.dumps({'x': timestamps,
                'y': price_data})
    return jsonify(fig_json)

@app.route('/update_news', methods=['POST'])
def update_news():
    selected_currency = request.json.get('currency')

    data = News.query.filter_by(
        ticker=selected_currency
    ).with_entities(News.title, News.url, News.summary, News.overall_sentiment_score, News.overall_sentiment_label, News.ticker_relevance_score, News.ticker_sentiment_score, News.ticker_sentiment_label, News.time_published).order_by(desc(News.time_published)).all()

    # Send data as json

    data_dict = {
        'title': [row[0] for row in data],
        'url': [row[1] for row in data],
        'summary': [row[2] for row in data],
        'overall_sentiment_score': [row[3] for row in data],
        'overall_sentiment_label': [row[4] for row in data],
        'ticker_relevance_score': [row[5] for row in data],
        'ticker_sentiment_score': [row[6] for row in data],
        'ticker_sentiment_label': [row[7] for row in data],
        'time_published': [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in (row[8] for row in data)]
    }
    data_json = json.dumps(data_dict)
    # Send data as JSON
    return jsonify(data_json)