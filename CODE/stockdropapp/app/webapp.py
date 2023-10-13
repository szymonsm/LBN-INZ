from flask import Flask
# from webapp.views import *

app = Flask(__name__)
#from ews import *

if __name__ == '__main__':
    app.run(debug=True)