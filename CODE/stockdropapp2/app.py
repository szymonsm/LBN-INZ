from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sda_db.db'  # Change to your database URL
db = SQLAlchemy(app)
migrate = Migrate(app, db)

from views import *