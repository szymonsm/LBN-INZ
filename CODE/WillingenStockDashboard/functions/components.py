from dash import html
import pandas as pd
import dash_bootstrap_components as dbc
import math

def round_to_second_significant_digit(number):
    if number == 0:
        return 0.0
    else:
        return round(number, -int(math.floor(math.log10(abs(number)))) + 1)

def get_raw_news(symbol):
    url = f'https://raw.githubusercontent.com/szymonsm/wjbvouvow480f2cwuvbog0/main/{symbol}/raw_news.csv'
    df = pd.read_csv(url)
    return df

def get_finance(symbol):
    url = f'https://raw.githubusercontent.com/szymonsm/wjbvouvow480f2cwuvbog0/main/{symbol}/raw_finance.csv'
    df = pd.read_csv(url)
    return df

def get_processed(symbol):
    url = f'https://raw.githubusercontent.com/szymonsm/wjbvouvow480f2cwuvbog0/main/{symbol}/processed.csv'
    df = pd.read_csv(url)
    return df

def get_predictions(symbol):
    url = f'https://raw.githubusercontent.com/szymonsm/wjbvouvow480f2cwuvbog0/main/{symbol}/predictions.csv'
    df = pd.read_csv(url)
    return df

# Function to calculate KPIs (replace with actual calculations)
def calculate_kpis(processed, marker):
    # calculate mean difference between actual and predicted values
    mean_diff_1 = round_to_second_significant_digit(abs(processed[f'{marker}_Close'] - processed[f'{marker}_Close'].shift(1)).dropna().mean())
    if marker == 'BTC-USD':
        mean_diff_5 = round_to_second_significant_digit(abs(processed[f'{marker}_Close'] - processed[f'{marker}_Close'].shift(7)).dropna().mean())
    else:
        mean_diff_5 = round_to_second_significant_digit(abs(processed[f'{marker}_Close'] - processed[f'{marker}_Close'].shift(5)).dropna().mean())
    min_close = round(processed[f'{marker}_Close'].min(),2)
    mean_close = round(processed[f'{marker}_Close'].mean(),2)
    max_close = round(processed[f'{marker}_Close'].max(),2)
    mean_future = round_to_second_significant_digit(processed['mean_future'].mean())
    mean_influential = round_to_second_significant_digit(processed['mean_influential'].mean())
    mean_trustworthy = round_to_second_significant_digit(processed['mean_trustworthy'].mean())
    mean_clickbait = round_to_second_significant_digit(processed['mean_clickbait'].mean())
    mean_sentiment = round_to_second_significant_digit(processed['finbert_Score'].mean())
    card_mean_diff_1 = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{mean_diff_1}",
                    className="card-value",
                ),
                html.P(
                    "MDIP1",
                    className="card-target",
                ),
            ])
        ])

    card_mean_diff_5 = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{mean_diff_5}",
                    className="card-value",
                ),
                html.P(
                    "MDIP5",
                    className="card-target",
                ),
            ])
        ])
    
    card_min_close = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{min_close}",
                    className="card-value",
                ),
                html.P(
                    "Min Close",
                    className="card-target",
                ),
            ])
        ])

    card_mean_close = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{mean_close}",
                    className="card-value",
                ),
                html.P(
                    "Mean Close",
                    className="card-target",
                ),
            ])
        ])
    
    card_max_close = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{max_close}",
                    className="card-value",
                ),
                html.P(
                    "Max Close",
                    className="card-target",
                ),
            ])
        ])
    
    card_mean_future = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{mean_future}",
                    className="card-value",
                ),
                html.P(
                    "Mean Future",
                    className="card-target",
                ),
            ])
        ])
    
    card_mean_influential = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{mean_influential}",
                    className="card-value",
                ),
                html.P(
                    "Mean Influential",
                    className="card-target",
                ),
            ])
        ])
    
    card_mean_trustworthy = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{mean_trustworthy}",
                    className="card-value",
                ),
                html.P(
                    "Mean Trustworthy",
                    className="card-target",
                ),
            ])
        ])

    card_mean_clickbait = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{mean_clickbait}",
                    className="card-value",
                ),
                html.P(
                    "Mean Clickbait",
                    className="card-target",
                ),
            ])
        ])
    
    card_mean_sentiment = dbc.Card([
        dbc.CardBody([
                html.P(
                    f"{mean_sentiment}",
                    className="card-value",
                ),
                html.P(
                    "Mean Sentiment",
                    className="card-target",
                ),
            ])
        ])

    return card_mean_diff_1, card_mean_diff_5, card_min_close, card_mean_close, card_max_close, card_mean_future, card_mean_influential, card_mean_trustworthy, card_mean_clickbait, card_mean_sentiment

# Function to load all data
def get_stock_data_all(MARKERS):
    data = dict()
    for marker in MARKERS:
        data[marker] = dict()
        data[marker]['processed'] = get_processed(marker)
        data[marker]['finance'] = get_finance(marker)
        data[marker]['news'] = get_raw_news(marker)
        data[marker]['predictions'] = get_predictions(marker)
    return data