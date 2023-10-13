from flask import Flask

app = Flask(__name__)
# app.config.from_object('config')

# Import views and models
from app import views, models