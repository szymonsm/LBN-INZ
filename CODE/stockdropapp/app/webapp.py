from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'  # Change to your database URL
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# if __name__ == '__main__':
#     app.run(debug=True)