import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


app = Flask(__name__)
model = pickle.load(open('nlp.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text_features =  [x for x in request.form.values() ]
    final_features = ''.join(text_features)
    prediction = model(final_features).cats['DOCUMENTATION']

    output = round(prediction, 4)

    return render_template('index.html', prediction_text='Documentation Prediction Score is {}'.format(output))
    


if __name__ == "__main__":
    app.run(debug=True)
