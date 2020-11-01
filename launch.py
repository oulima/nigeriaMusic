
from flask import Flask, render_template, request
import json
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from stop_words import get_stop_words
import nltk
import re
from unidecode import unidecode

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fonctions import prediction, entrainement, nettoyage

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", title='Home')


@app.route("/prediction",methods=['POST'])
def text():
    user_text = request.form.get('input_text')
    retour=prediction(user_text)
    return render_template("interface.html", input_text=user_text,prediction=retour)




@app.route("/entrainement",methods=['GET'])
def entr(usertexte=None):
    retour = entrainement()
    #return json.dumps({'text_user':retour})
    return render_template("entrainement.html",entrainement=retour) 

""" @app.route("/entrainement",methods=['GET'])
def entr():
    retour = pd.read_csv('.Nigeria Music/nigerian.csv')

    return json. dumps({'response':'Le musicien est {} et il a une popularité de: {}'.format(score['artist'], score['popularity'])})    """






if __name__ == "__main__":
    app.run(debug=True)

