from app import app, db
from flask import render_template, request, redirect, url_for, jsonify
from models import StockPrice, News
from sqlalchemy import inspect, desc
from datetime import datetime
import yaml
import json
import pandas as pd
from datetime import timedelta
from code.alphavantage_stock_price_downloader import AlphaVantageStockPriceDownloader
from code.marketaux_news_collector import MarketauxNewsDownloader
from code.yahoofinanse_downloader import YFinanceDownloader

running_daemons = {}

@app.route('/')
def index():
    print('index')
    with open('config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    currency_options = config['options']['currency']
    time_period_options = config['options']['time_period']
    return render_template('index.html', currency_options=currency_options, time_period_options=time_period_options, running_daemons=running_daemons)

@app.route('/download_data/<type>/<ticker>/<month>')
def download_data(type, ticker, month):
    # Get user input (e.g., ticker, month, interval)

    with open('config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    #interval = int(interval)

    #api_keys = config['api_keys']['alphavantage']

    if type == 'stock_price':
        # Use your data downloader class to fetch data (replace with actual code)
        # check if begin_date is greater than end_date
        if datetime.strptime(month[:8], "%Y%m%d") >= datetime.strptime(month[8:16], "%Y%m%d"):
            raise ValueError("End date must be greater than begin date")
        if ticker == "BTCUSD":
            downloader = YFinanceDownloader(["BTC-USD"], month[:8], month[8:16])
            df = downloader.create_df().rename(columns={f"BTC-USD_Open":"Open", f"BTC-USD_High":"High", f"BTC-USD_Low":"Low", f"BTC-USD_Close":"Close", f"BTC-USD_Adj Close":"Adj Close", f"BTC-USD_Volume":"Volume"})
        else:
            downloader = YFinanceDownloader([ticker], month[:8], month[8:16])
            df = downloader.create_df().rename(columns={f"{ticker}_Open":"Open", f"{ticker}_High":"High", f"{ticker}_Low":"Low", f"{ticker}_Close":"Close", f"{ticker}_Adj Close":"Adj Close", f"{ticker}_Volume":"Volume"})

        df['Ticker'] = ticker
        # df = downloader.download_intraday_ticker_data(month, interval)
        date_format = '%Y-%m-%d'

        # Store the data in the database
        for _, row in df.iterrows():
            stock_price = StockPrice(
                date=row['Date'],
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                adj_close=row['Adj Close'],
                volume=row['Volume'],
                ticker=ticker,
            )
            # check if stock_price already exists
            if StockPrice.query.filter_by(date=row['Date'], ticker=ticker).first() is None:
                db.session.add(stock_price)
        db.session.commit()

    elif type == 'news':
        api_keys = config['api_keys']['marketaux']
        begin_date = month[:4] + '-' + month[4:6] + '-' + month[6:8]
        end_date = month[8:12] + '-' + month[12:14] + '-' + month[14:16]
        # check if end_date is greater than begin_date
        if datetime.strptime(end_date, "%Y-%m-%d") <= datetime.strptime(begin_date, "%Y-%m-%d"):
            raise ValueError("End date must be greater than begin date")
        downloader = MarketauxNewsDownloader(api_keys, ticker, begin_date, end_date)
        df = pd.DataFrame.from_dict(downloader.download_multiple_data(max_num_requests=100, pages=[1, 2, 3]))
        while downloader.end_date < datetime.strptime(end_date, "%Y-%m-%d"):
            downloader.begin_date = downloader.end_date + timedelta(days=1)
            downloader.end_date = datetime.strptime(end_date, "%Y-%m-%d")
            dict_news = downloader.download_multiple_data(max_num_requests=100, pages=[1, 2, 3])
            df = pd.concat([df, pd.DataFrame.from_dict(dict_news)])
            #df = df.concat(pd.DataFrame.from_dict(dict_news))
        # date_format = '%Y%m%dT%H%M%S'
        # 2022-03-25T19:37:56.000000Z
        date_format = '%Y-%m-%dT%H:%M:%S.000000Z'
        for _, row in df.iterrows():
            # uuid,title,description,keywords,snippet,url,image_url,language,published_at,source,relevance_score,type,industry,match_score,sentiment_score,ticker
            news = News(
                uuid=row['uuid'],
                title=row['title'],
                description=row['description'],
                keywords=row['keywords'],
                snippet=row['snippet'],
                url=row['url'],
                image_url=row['image_url'],
                language=row['language'],
                published_at=datetime.strptime(row['published_at'], date_format),
                source=row['source'],
                relevance_score=row['relevance_score'],
                type=row['type'],
                industry=row['industry'],
                match_score=row['match_score'],
                sentiment_score=row['sentiment_score'],
                # summary=row['summary'],
                # overall_sentiment_score=row['overall_sentiment_score'],
                # overall_sentiment_label=row['overall_sentiment_label'],
                # ticker_relevance_score=row['ticker_relevance_score'],
                # ticker_sentiment_score=row['ticker_sentiment_score'],
                # ticker_sentiment_label=row['ticker_sentiment_label'],
                # time_published=datetime.strptime(row['time_published'], date_format),
                ticker=ticker
            )
            # check if news already exists
            if News.query.filter_by(uuid=row['uuid']).first() is None:
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
    # sort news by time_published
    news = sorted(news, key=lambda x: x.published_at, reverse=True)
    columns_news = [column.key for column in inspect(News).c]
    return render_template('display_tables.html', stock_prices=stock_prices, news=news, columns_prices=columns_prices, columns_news=columns_news)

@app.route('/update_plots', methods=['POST'])
def update_plot():
    selected_currency = request.json.get('currency')
    selected_time_period = request.json.get('time_period')
    selected_data_type = request.json.get('data_type')

    # Modify the query to filter data based on selected criteria
    # As for now I am sending all the data to the frontend
    # Take Date and Close columns from the database
    
    data = StockPrice.query.filter_by(
        ticker=selected_currency,
    ).with_entities(StockPrice.date, StockPrice.close).all()

    # Need of sending timestamps as strings to the frontend
    timestamps = [str(row[0]) for row in data]
    price_data = [row[1] for row in data]
    fig_json = json.dumps({'x': timestamps,
                'y': price_data})
    return jsonify(fig_json)

@app.route('/update_news', methods=['POST'])
def update_news():
    selected_currency = request.json.get('currency')
    # uuid,title,description,keywords,snippet,url,image_url,language,published_at,source,relevance_score,type,industry,match_score,sentiment_score,ticker

    data = News.query.filter_by(
        ticker=selected_currency
    ).with_entities(News.uuid, News.title, News.description, News.keywords, News.snippet, News.url, News.image_url, News.language, News.published_at, News.source, News.relevance_score, News.type, News.industry, News.match_score, News.sentiment_score, News.ticker).order_by(desc(News.published_at)).all()
    #.with_entities(News.title, News.url, News.summary, News.overall_sentiment_score, News.overall_sentiment_label, News.ticker_relevance_score, News.ticker_sentiment_score, News.ticker_sentiment_label, News.time_published).order_by(desc(News.time_published)).all()

    # Send data as json

    data_dict = {
        'uuid': [row[0] for row in data],
        'title': [row[1] for row in data],
        'description': [row[2] for row in data],
        'keywords': [row[3] for row in data],
        'snippet': [row[4] for row in data],
        'url': [row[5] for row in data],
        'image_url': [row[6] for row in data],
        'language': [row[7] for row in data],
        'published_at': [row[8].strftime('%Y-%m-%d %H:%M:%S') for row in data],
        'source': [row[9] for row in data],
        'relevance_score': [row[10] for row in data],
        'type': [row[11] for row in data],
        'industry': [row[12] for row in data],
        'match_score': [row[13] for row in data],
        'sentiment_score': [row[14] for row in data],
        'ticker': [row[15] for row in data]
        # 'title': [row[0] for row in data],
        # 'url': [row[1] for row in data],
        # 'summary': [row[2] for row in data],
        # 'overall_sentiment_score': [row[3] for row in data],
        # 'overall_sentiment_label': [row[4] for row in data],
        # 'ticker_relevance_score': [row[5] for row in data],
        # 'ticker_sentiment_score': [row[6] for row in data],
        # 'ticker_sentiment_label': [row[7] for row in data],
        # 'time_published': [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in (row[8] for row in data)]
    }
    data_json = json.dumps(data_dict)
    # Send data as JSON
    return jsonify(data_json)