#**************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import os

from model import Algo

#***************** FLASK *****************************
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        name = f.filename
        obj = Algo(name)
        result = obj.cnn_algo()
        print(result)
        return result

    return None

if __name__ == '__main__':
   app.run(debug=True)
