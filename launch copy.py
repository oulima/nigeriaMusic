
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

FR = SnowballStemmer('french')
MY_STOP_WORD_LIST = get_stop_words('french')
FINAL_STOPWORDS_LIST = stopwords.words('french')

S_W = list(set(FINAL_STOPWORDS_LIST + MY_STOP_WORD_LIST))
S_W = [elem.lower() for elem in S_W]

CLS = pickle.load(open("./cls.pkl", "rb"))
LOADED_VEC = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("./cls.pkl", "rb")))

VECTORIZER = TfidfVectorizer()


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", title='Home')



@app.route("/entrainemen",methods=['GET'])
def entr():
    retour = pd.read_csv('.Nigeria Music/nigerian.csv')
    result = doTraining(data)
    return json.dumps({'response':'Le musicien est {} et il a une popularité de: {}'.format(result['artist'], result['popularity'])})



@ app.route ( "/predictio" , methods = [ 'POST' ])
def  retour ():
    
    user_text  =  request.form.get ( 'input_text' )
    lr_baseline  =  cornichon.load(open('./cls.pkl', 'rb' ))
    resp  =  int (lr_baseline.prédire(np.tableau ([ int ( user_text ), 4 ]). reshape ( 1 , - 1 )))

    retour  "L'estimation de l'artiste est de:"  +  str ( resp ) +  "$"







def getPrediction(user_text):
    transformer = TfidfTransformer()
    user = transformer.fit_transform(LOADED_VEC.fit_transform([nettoyage(user_text)]))
    if CLS.predict(user)[0].astype(str) == '1':
        valeurText = "populaire"
    else:
        valeurText = "inpopulaire"
    proba = round(CLS.predict_proba(user).max(), 2) * 100
    return {
        "valeurText": valeurText,
        "proba": proba
    }


def doTraining(retour):
    x = vectorisation(retour['name'].apply(nettoyage))
    pickle.dump(VECTORIZER.vocabulary_, open("./cls.pkl", "wb"))

    y = retour['label']

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    cls = LogisticRegression(max_iter=300).fit(x_train, y_train)
    pickle.dump(cls, open("./cls.pkl", "wb"))
    CLS = pickle.load(open("./cls.pkl", "rb"))
    LOADED_VEC = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("./cls.pkl", "rb")))
    return {
        "artist": round(cls.score(x_val, y_val), 2) * 100,
        "popularity": len(retour['name'])
    }
def nettoyage(string):
    l = []
    string = unidecode(string.lower())
    string = " ".join(re.findall("[a-zA-Z]+", string))

    for word in string.split():
        if word in S_W:
            continue
        else:
            l.append(FR.stem(word))
    return ' '.join(l)

def vectorisation(text):
    VECTORIZER.fit(text)
    return VECTORIZER.transform(text)

if __name__ == "__main__":
    #app.run(host = '0.0.0.0', port = 80)
    app.run(debug=True)