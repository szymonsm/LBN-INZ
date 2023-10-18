import yaml
from flask import render_template
from app import app
from app.daemon_manager import start_daemon, stop_daemon

# Store running daemons
running_daemons = {}

@app.route('/')
def index():
    
    with open('app/config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    currency_options = config['options']['currency']
    time_period_options = config['options']['time_period']
    return render_template('index.html', currency_options=currency_options, time_period_options=time_period_options, running_daemons=running_daemons)


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
