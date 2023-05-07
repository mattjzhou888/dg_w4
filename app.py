import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open("model.pickle", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    X = np.array([[int(x) for x in request.form.values()]])
    pred = model.predict(X)
    out = round(pred[0], 2)

    return render_template("index.html", prediction_text="Predicted Salary: {}".format(out))

app.run(host='0.0.0.0', port=50000)