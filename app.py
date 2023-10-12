from flask import Flask , request
from algrithm import getPrediction
app = Flask(__name__)


@app.route('/get', methods = ['GET'])
def get_predict(age=None,gender=None,region=None):

    print('get in prediction')
    gender = request.values['gender']
    age = request.values['age']
    region = request.values['region']

    print(gender)
    print(age)
    print(region)

    return getPrediction(age,gender,region)


@app.route('/lc')
def get_learning_curve():
    return get_learning_curve()



if __name__ == '__main__':
    app.run()
