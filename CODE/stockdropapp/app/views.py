import yaml
from flask import render_template
from app import app

@app.route('/')
def index():
    
    with open('app/config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    currency_options = config['options']['currency']
    time_period_options = config['options']['time_period']
    return render_template('index.html', currency_options=currency_options, time_period_options=time_period_options)
    # return render_template('index1.html')

# from app import app
# from flask import render_template

# @app.route('/')
# def index():
#     return render_template('index1.html')
