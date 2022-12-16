from flask import Flask
from algrithm import getPrediction
app = Flask(__name__)


@app.route('/')
def get_predict():  # put application's code here
    return getPrediction()


if __name__ == '__main__':
    app.run()
