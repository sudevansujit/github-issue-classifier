import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from pathlib import Path, PureWindowsPath, PurePath, PurePosixPath, WindowsPath, PosixPath
fname = "nlp.pkl"
# fpath = PureWindowsPath(fname)
fpath = Path(fname)



# import pathlib
# plt = platform.system()
# if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
with open(fpath, "rb") as fd:
# Works in Winbug$ only
    model = pkl.load(fd)
# model = pickle.load(open('nlp.pkl', 'rb'))

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
