# import os

# class Config:
#     # Set a secret key for your application (for sessions, etc.)
#     SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key_here'

#     # Configure the database URL (SQLite in this example)
#     SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///your_database.db'
#     SQLALCHEMY_TRACK_MODIFICATIONS = False  # Turn off modification tracking for SQLAlchemy

#     # Enable SQLAlchemy echo mode for debugging (set to False in production)
#     SQLALCHEMY_ECHO = True

#     # Path to your migrations directory for Flask-Migrate
#     SQLALCHEMY_MIGRATE_REPO = os.path.join(os.path.dirname(__file__), 'migrations')

# # You can create different configurations for development, production, etc.
# class DevelopmentConfig(Config):
#     DEBUG = True

# class ProductionConfig(Config):
#     DEBUG = False

# # Add more configurations as needed

# # Choose the configuration based on an environment variable (FLASK_ENV)
# config = {
#     'development': DevelopmentConfig,
#     'production': ProductionConfig,
#     # Add more configurations here
# }

# # Set the active configuration based on the FLASK_ENV environment variable
# config_name = os.environ.get('FLASK_ENV', 'development')
# app_config = config[config_name]
