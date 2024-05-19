from flask import Flask, request, jsonify
import pickle
from perceptron import Perceptron
import numpy as np

# Create a flask
app = Flask(__name__)


# Create an API end point
@app.route('/predict_get', methods=['GET'])
def get_prediction():
    # sepal length
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))
    
    features = [sepal_length, petal_length]

    # Load pickled model file
    with open('model.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)
        
    # Predict the class using the model
    predicted_class = int(model.predict(features))
    
    # Return a json object containing the features and prediction
    return jsonify(features=features, predicted_class=predicted_class)

@app.route('/predict_post', methods=['POST'])
def post_predict():
    data = request.get_json(force=True)
    # sepal length
    sepal_length = float(data.get('sl'))
    petal_length = float(data.get('pl'))
    
    features = [sepal_length, petal_length]

    # Load pickled model file
    with open('model.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)
        
    # Predict the class using the model
    predicted_class = int(model.predict(features))
    output = dict(features=features, predicted_class=predicted_class)
    # Return a json object containing the features and prediction
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
