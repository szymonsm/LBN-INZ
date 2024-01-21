import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from dash import dash_table
import dash_bootstrap_components as dbc
from dateutil.relativedelta import relativedelta
import math
import plotly.graph_objects as go
from functions.components import calculate_kpis, get_stock_data_all

MARKERS = ['BA', 'BTC-USD', 'NFLX', 'TSLA']
start_date_str = '202301'
start_date = datetime.strptime(start_date_str, '%Y%m')

# Initialize an empty list to store the results for possible dates in a slider
DATE_LIST = []

# Generate dates for the next 100 years to be used in the slider
for _ in range(4000):  # 100 years * 12 months
    DATE_LIST.append(start_date.strftime('%Y%m'))
    start_date += timedelta(weeks=1)  # Increment by one week
DATE_LIST = list(sorted(set([int(x) for x in DATE_LIST])))

# Load all data and store it in a dictionary
# Set the initial date range to be the minimum and maximum dates in the data
DATA = get_stock_data_all(MARKERS)
mind = DATA['BA']['processed']['Date'].min()
maxd = DATA['BA']['processed']['Date'].max()
mind = datetime.strptime(mind, '%Y-%m-%d')
maxd = datetime.strptime(maxd, '%Y-%m-%d')
m = {DATE_LIST.index(int(mind.strftime('%Y%m'))): mind.strftime('%b %Y')}
cd = mind
while cd < maxd:
    cd += timedelta(days=32)
    m[int(cd.strftime('%Y%m'))] = cd.strftime('%b %Y')

# Initial setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME])
app.title = "🗻 Willingen Stock Dashboard 🗻"

# Layout of the app
app.layout = html.Div([
    # Title
    html.H1("🗻 Willingen Stock Dashboard 🗻", style={'textAlign': 'center','font-family': 'Monospace', 'font-weight': 'bold', 'color': 'rgba(181, 245, 255, 0.8)', 'padding': '10px'}),
    
    # Dropdown to select stock symbol
    html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='stock-dropdown',
                    options=[
                        {'label': 'Boeing (BA)', 'value': 'BA'},
                        {'label': 'Bitcoin (BTC-USD)', 'value': 'BTC-USD'},
                        {'label': 'Netflix (NFLX)', 'value': 'NFLX'},
                        {'label': 'Tesla (TSLA)', 'value': 'TSLA'}
                    ],
                    value='BA',  # Default value
                    clearable=False,
                    searchable=False,
                    style={'width': '75%',
                            'margin': '0 auto',
                            'background-color': 'rgba(0, 0, 0, 0)',
                            'color': 'rgba(181, 245, 255, 0.8)',
                            'border-radius': '10px',
                            'box-shadow': 'rgba(181, 245, 255, 0.8)',
                            'text-align': 'center',
                            'background': 'rgba(0, 0, 0, 0)',
                            'listbox': 'rgba(0, 0, 0, 0)',
                    }
                ),
                # Rows of KPIs
                html.Div(id='row1', className='row'),
                html.Div(id='row2', className='row'),
                html.Div(id='row3', className='row'),
                html.Div(id='row4', className='row'),
            ], width=5),
            dbc.Col([
                # Date slider
                dcc.RangeSlider(
                    id='date-slider',
                    min=DATE_LIST.index(int(mind.strftime('%Y%m'))),
                    max=DATE_LIST.index(int(maxd.strftime('%Y%m'))),
                    step=1,
                    value=[DATE_LIST.index(int(mind.strftime('%Y%m'))), DATE_LIST.index(int(maxd.strftime('%Y%m')))],
                    marks=m,
                    vertical=False,
                    updatemode='mouseup',
                ),
                # Candlestick chart
                dcc.Graph(id='price-chart', style={'height': '40vh'}),
                # Sentiment scores chart
                dcc.Graph(id='news-chart', style={'height': '40vh'}),
            ], width=7),
        ]),
        dbc.Row([
            # Prediction chart
            dbc.Col([dcc.Graph(id='prediction-chart', style={'height': '40vh'})]),
        ]),
        dbc.Row([
            dbc.Col([], width=2),
            # Confusion matrix based on predictions
            dbc.Col([dcc.Graph(id='indicator-chart'),], width=4),
            # Top 3 most active websites for selected stock
            dbc.Col([dcc.Graph(id='top-websites-chart'),], width=4),
            dbc.Col([], width=2),
        ]),
        dbc.Row([
            # Table with last 5 news of selected stock
            dbc.Col([html.H2('Last 5 news', className='news-table-title'),
                    html.Div(id='news-table-container', className='news-table-container')], width=12),
        ]),

    ], className='container'),

    html.Div([
        html.Hr(),
        # Footer
        html.P('Created by Wojciech Kosiuk, Szymon Matuszewski and Michał Mazuryk', style={'textAlign': 'center','font-family': 'Monospace', 'font-weight': 'bold', 'color': 'rgba(181, 245, 255, 0.8)'}),
        html.P('Warsaw University of Technology', style={'textAlign': 'center','font-family': 'Monospace', 'font-weight': 'bold', 'color': 'rgba(181, 245, 255, 0.8)'}),
    ]),
    
])


# Callbacks to create slider based on available dates
@app.callback(
    [Output('date-slider', 'marks'),
     Output('date-slider', 'min'),
     Output('date-slider', 'max'),
     Output('date-slider', 'value')],
    Input('stock-dropdown', 'value')
)
def update_slider(selected_stock):
    """
    Function to update slider based on available dates for selected stock

    Parameters
    ----------
    selected_stock : str
        Selected stock symbol

    Returns
    -------
    marks : dict
        Dictionary with marks for the slider
    min_ : int
        Index of the minimum date in DATE_LIST
    max_ : int
        Index of the maximum date in DATE_LIST
    value : list
        List with index of the minimum and maximum date in DATE_LIST
    """
    # Get stock data
    stock_data = DATA[selected_stock]['processed']
    # Prepare min and max dates
    min_date = stock_data['Date'].min()
    max_date = stock_data['Date'].max()
    if isinstance(min_date, str):
        min_date = datetime.strptime((min_date), '%Y-%m-%d')
        max_date = datetime.strptime((max_date), '%Y-%m-%d')
    # Initialize marks
    marks = {DATE_LIST.index(int(min_date.strftime('%Y%m'))): min_date.strftime('%b %Y')}
    # Increment date by one month until the end date is reached
    current_date = min_date
    while current_date < max_date:
        current_date += timedelta(days=32)
        # Add current date to marks
        marks[DATE_LIST.index(int(current_date.strftime('%Y%m')))] = current_date.strftime('%b %Y')
    # Get index of min and max dates
    min_=DATE_LIST.index(int(min_date.strftime('%Y%m')))
    max_=DATE_LIST.index(int(max_date.strftime('%Y%m')))
    # Set value to be the min and max dates
    value=[min_, max_]
    return (marks, min_, max_, value)

# Callbacks to update data based on user inputs
@app.callback(
    [Output('news-table-container', 'children'),
     Output('prediction-chart', 'figure'),
     Output('price-chart', 'figure'),
     Output('indicator-chart', 'figure'),
     Output('top-websites-chart', 'figure'),
     Output('news-chart', 'figure'),
     Output('row1', 'children'),
     Output('row2', 'children'), 
     Output('row3', 'children'),
     Output('row4', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('date-slider', 'value')]
)
def update_data(selected_stock, selected_dates):
    """
    Function to update data based on user inputs

    Parameters
    ----------
    selected_stock : str
        Selected stock symbol
    selected_dates : list
        List with index of the minimum and maximum date in DATE_LIST
        
    Returns
    -------
    news-table-container : dash_table.DataTable
        Table with last 5 news of selected stock
    prediction-chart : plotly.graph_objects.Figure
        Figure with prediction chart
    price-chart : plotly.graph_objects.Figure
        Figure with price chart
    indicator-chart : plotly.graph_objects.Figure
        Figure with indicator chart
    top-websites-chart : plotly.graph_objects.Figure
        Figure with top websites chart
    news-chart : plotly.graph_objects.Figure
        Figure with news chart
    row1 : html.Div
        Row with KPIs
    row2 : html.Div
        Row with KPIs
    row3 : html.Div
        Row with KPIs
    row4 : html.Div
        Row with KPIs
    """
    # Get start and end dates as strings in format YYYYMM
    start_date, end_date = str(DATE_LIST[selected_dates[0]]), str(DATE_LIST[selected_dates[1]])
    # Get current date as string in format YYYYMM
    current_date = datetime.now().strftime('%Y%m')
    if end_date == current_date:
        prediction_flag = True
    else:
        prediction_flag = False
    # Convert start and end dates to datetime objects
    sd = datetime.strptime(start_date, '%Y%m')
    start_date = sd.replace(day=1)
    ed = datetime.strptime(end_date, '%Y%m')
    end_date = ed.replace(day=1) + relativedelta(months=+1) - timedelta(days=1)
    
    # Get stock data between start and end dates
    processed = DATA[selected_stock]['processed']
    processed['Date'] = pd.to_datetime(processed['Date'])
    processed = processed[(processed['Date'] >= start_date) & (processed['Date'] <= end_date)]

    # Get predictions between start and end dates
    predictions = DATA[selected_stock]['predictions']
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    if prediction_flag:
        predictions = predictions[(predictions['Date'] >= start_date)]
    else:
        predictions = predictions[(predictions['Date'] >= start_date) & (predictions['Date'] <= end_date)]

    # Get financial news between start and end dates (here is the different format of dates, so we need to convert them to datetime objects)
    news = DATA[selected_stock]['news']

    if isinstance(news['published_at'][0], str):
        news['published_at'] = news['published_at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    news = news[(news['published_at'] >= start_date) & (news['published_at'] <= end_date)]
    news_tail = news.tail(5)
    # Sort by published_at in descending order
    news_tail = news_tail.sort_values(by='published_at', ascending=False).reset_index(drop=True)
    # Create data table with every column from news_tail
    news_list = []
    for index, row in news_tail.iterrows():
        news_list.append({
            '': index + 1,
            'title': row['title'],
            'description': row['description'],
            'snippet': row['snippet'],
            'published_at': row['published_at'],
            'source': row['source'],
            'type': row['type'],
            'industry': row['industry'],
            'match_score': round(row['match_score'],2),
            'sentiment_score': round(row['sentiment_score'],2),
        })    

    # Get finance data between start and end dates
    finance = DATA[selected_stock]['finance']
    finance['Date'] = pd.to_datetime(finance['Date'])
    finance = finance[(finance['Date'] >= start_date) & (finance['Date'] <= end_date)]

    # Calculate KPIs and apply the cards to selected rows in layout
    card_mean_diff_1, card_mean_diff_5, card_min_close, card_mean_close, card_max_close, card_mean_future, card_mean_influential, card_mean_trustworthy, card_mean_clickbait, card_mean_sentiment = calculate_kpis(processed, selected_stock)
    row1 = html.Div([
            dbc.Row([
                    dbc.Col([card_mean_diff_1]),
                    dbc.Col([card_mean_diff_5]),
                    
                ]),
        ], style={'padding': '10px'}
    )
    row2 = html.Div([
            dbc.Row([
                    dbc.Col([card_min_close]),
                    dbc.Col([card_mean_close]),
                    dbc.Col([card_max_close]),
                ]),
        ], style={'padding': '10px'}
    )
    row3 = html.Div([
            dbc.Row([
                    dbc.Col([card_mean_future]),
                    dbc.Col([card_mean_influential]),
                    dbc.Col([card_mean_trustworthy]),
                ]),
        ], style={'padding': '10px'}
    )
    row4 = html.Div([
            dbc.Row([
                    dbc.Col([card_mean_clickbait]),
                    dbc.Col([card_mean_sentiment]),
                ]),
        ], style={'padding': '10px'}
    )
    # Create prediction chart with 'Prediction' trace from predictions and 'target_5' trace from processed or 'target_7' if selected_stock is 'BTC-USD'
    if selected_stock == 'BTC-USD':
        prediction_chart = px.line(predictions, x='Date', y=['Prediction'], title=f'{selected_stock} Target Prediction')
        # Add target_7 trace
        prediction_chart.add_scatter(x=processed['Date'], y=processed['target_7'], mode='lines', name='target_7')
    else:
        prediction_chart = px.line(predictions, x='Date', y=['Prediction'], title=f'{selected_stock} Target Prediction')
        # Add target_5 trace
        prediction_chart.add_scatter(x=processed['Date'], y=processed['target_5'], mode='lines', name='target_5')
    # Update layout of prediction_chart
    prediction_chart.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis_title=f'{selected_stock} Price',
        xaxis_title='Date',
        title=f'{selected_stock} Target Prediction',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font_color='rgba(181, 245, 255, 0.8)',
    )

    # Create price_chart with candlestick chart from processed
    price_chart = go.Figure(go.Candlestick(
        x=processed['Date'],
        open=processed[f'{selected_stock}_Open'],
        high=processed[f'{selected_stock}_High'],
        low=processed[f'{selected_stock}_Low'],
        close=processed[f'{selected_stock}_Close']
    ))
    # Update layout of price_chart
    price_chart.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis_title=f'{selected_stock} Price',
        xaxis_title='Date',
        title=f'{selected_stock} Price Chart',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font_color='rgba(181, 245, 255, 0.8)',
    )

    # Join predictions and processed['target_5'] or processed['target_7'] if selected_stock is 'BTC-USD' and remove rows with any NaN value
    # Create confusion matrix based on predictions and processed['target_5'] or processed['target_7'] if selected_stock is 'BTC-USD'
    if selected_stock == 'BTC-USD':
        predictions_joined = predictions.join(processed['target_7']).dropna()
        matrix = [[0, 0], [0, 0]]
        for index, row in predictions_joined.iterrows():
            if row['Prediction'] > 0 and row['target_7'] > 0:
                matrix[0][0] += 1
            elif row['Prediction'] < 0 and row['target_7'] > 0:
                matrix[0][1] += 1
            elif row['Prediction'] > 0 and row['target_7'] < 0:
                matrix[1][0] += 1
            elif row['Prediction'] < 0 and row['target_7'] < 0:
                matrix[1][1] += 1
    else:
        predictions_joined = predictions.join(processed['target_5']).dropna()
        matrix = [[0, 0], [0, 0]]
        for index, row in predictions_joined.iterrows():
            if row['Prediction'] > 0 and row['target_5'] > 0:
                matrix[0][0] += 1
            elif row['Prediction'] < 0 and row['target_5'] > 0:
                matrix[0][1] += 1
            elif row['Prediction'] > 0 and row['target_5'] < 0:
                matrix[1][0] += 1
            elif row['Prediction'] < 0 and row['target_5'] < 0:
                matrix[1][1] += 1

    # Create indicator_chart with created matrix
    indicator_chart = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Positive', 'Negative'],
        y=['Negative', 'Positive'],
        colorscale=[
                    # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
                    [0, "rgb(0, 0, 0)"],
                    [0.1, "rgb(0, 0, 0)"],

                    # Let values between 10-20% of the min and max of z
                    [0.1, "rgb(20, 20, 20)"],
                    [0.2, "rgb(20, 20, 20)"],

                    # Values between 20-30% of the min and max of z
                    [0.2, "rgb(40, 40, 40)"],
                    [0.3, "rgb(40, 40, 40)"],

                    [0.3, "rgb(60, 60, 60)"],
                    [0.4, "rgb(60, 60, 60)"],

                    [0.4, "rgb(80, 80, 80)"],
                    [0.5, "rgb(80, 80, 80)"],

                    [0.5, "rgb(100, 100, 100)"],
                    [0.6, "rgb(100, 100, 100)"],

                    [0.6, "rgb(120, 120, 120)"],
                    [0.7, "rgb(120, 120, 120)"],

                    [0.7, "rgb(140, 140, 140)"],
                    [0.8, "rgb(140, 140, 140)"],

                    [0.8, "rgb(160, 160, 160)"],
                    [0.9, "rgb(160, 160, 160)"],

                    [0.9, "rgb(180, 180, 180)"],
                    [1.0, "rgb(180, 180, 180)"]
    ],
    ))
    # Prepare annotations for the heatmap
    annotations = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(matrix[i][j]),
                    showarrow=False,
                    font=dict(
                        color='rgba(181, 245, 255, 0.8)',
                        size=20
                    )
                )
            )
    # Update layout of indicator_chart
    indicator_chart.update_layout(
        title=f'Model Confusion Matrix',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font_color='rgba(181, 245, 255, 0.8)',
        annotations=annotations,
    )

    # Take top 3 websites in case of counts in news and create a bar plot where 1 bar is silver, 2 bar is gold and 3 bar is bronze
    most_active_website = news['source'].value_counts().index[0]
    most_active_website_count = news['source'].value_counts().values[0]
    second_most_active_website = news['source'].value_counts().index[1]
    second_most_active_website_count = news['source'].value_counts().values[1]
    third_most_active_website = news['source'].value_counts().index[2]
    third_most_active_website_count = news['source'].value_counts().values[2]
    top_websites = pd.DataFrame({
        'Website': [second_most_active_website, most_active_website, third_most_active_website],
        'Count': [second_most_active_website_count, most_active_website_count, third_most_active_website_count]
    })
    top_websites_chart = px.bar(top_websites, x='Website', y='Count', title='Top 3 websites')
    # set_layout for firts bar to silver, second to gold and third to bronze
    top_websites_chart.update_traces(marker_color=['silver', 'gold', 'brown'])
    # Remove background and display Count over the bars (annotations)
    top_websites_chart.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)', annotations=[dict(x=xi, y=yi, text=str(yi), xanchor='center', yanchor='bottom', showarrow=False) for xi, yi in zip(top_websites['Website'], top_websites['Count'])], font_color='rgba(181, 245, 255, 0.8)',)

    # Create news chart with finbert_Score, vader_Score, bart_Score over time from processed
    news_chart = go.Figure()
    news_chart.add_trace(go.Scatter(x=processed['Date'], y=processed['finbert_Score'], mode='lines', name='finbert_Score'))
    news_chart.add_trace(go.Scatter(x=processed['Date'], y=processed['vader_Score'], mode='lines', name='vader_Score'))
    news_chart.add_trace(go.Scatter(x=processed['Date'], y=processed['bart_Score'], mode='lines', name='bart_Score'))
    # Update layout of news_chart
    news_chart.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis_title='Score',
        xaxis_title='Date',
        title='News Sentiment Scores',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font_color='rgba(181, 245, 255, 0.8)',
    )
    
    # Return all components
    return (
        dash_table.DataTable(
            id='news-table',
            columns=[{"name": col, "id": col} for col in news_list[0].keys()],
            data=news_list,
            style_table={
                'overflowX': 'auto',
                'width': '100%',
                'backgroundColor': 'rgba(0, 0, 0, 0)',
                'color': 'rgba(181, 245, 255, 0.8)',
            },
            style_cell={
                'textAlign': 'left',
                'whiteSpace': 'normal',
                'height': 'auto',
                'lineHeight': '15px',
                'backgroundColor': 'rgba(0, 0, 0, 0)',
            },
        ),
        prediction_chart,
        price_chart,
        indicator_chart,
        top_websites_chart,
        news_chart,
        row1,
        row2,
        row3,
        row4
    )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)