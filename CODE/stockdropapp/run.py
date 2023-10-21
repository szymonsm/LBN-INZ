from app import app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/szymo/LBN-INZ/CODE/stockdropapp/your_database.db'  # Change to your database URL
db = SQLAlchemy(app)
migrate = Migrate(app, db)

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/szymo/LBN-INZ/CODE/stockdropapp/your_database.db'  # Change to your database URL
# db = SQLAlchemy(app)
# migrate = Migrate(app, db)

# if __name__ == '__main__':
#     app.run(debug=True)