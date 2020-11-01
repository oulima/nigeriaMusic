import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
import pickle
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from stop_words import get_stop_words
import nltk
import re
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


my_stop_word_list = get_stop_words('french')
final_stopwords_list = stopwords.words('french')

s_w=list(set(final_stopwords_list+my_stop_word_list))
s_w=[elem.lower() for elem in s_w]

fr = SnowballStemmer('french')

#le corpus
df=pd.read_csv('nigerian.csv')
df['label']=df['popularity']
positif=df[df['label']>3].sample(226)
negatif=df[df['label']<3]
Corpus=pd.concat([positif,negatif],ignore_index=True)[['artist','label']]

for ind in Corpus['label'].index:
    if Corpus.loc[ind,'label'] > 3:
        Corpus.loc[ind,'label']=1
    elif Corpus.loc[ind,'label'] < 3:
        Corpus.loc[ind,'label']=0



def nettoyage(string):
   
    l=[]
    string=unidecode(string.lower())
    string=" ".join(re.findall("[a-zA-Z]+", string))
    
    for word in string.split():
        if word in s_w:
           continue
        else:
            l.append(fr.stem(word))
    return ' '.join(l)

Corpus['artist_net']=Corpus['artist'].apply(nettoyage)


def prediction(param):
   
   param=np.array([range(1,210,1712)]).reshape(1,-1)
   cls=pickle.load(open("cls.pkl", "rb"))
   return (cls.predict(param))


  
def entrainement():
    vectorizer = TfidfVectorizer()
    vectorizer.fit(Corpus['artist_net'])
    X=vectorizer.transform(Corpus['artist_net'])
    #sauver le voabulaire 
    pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

    y=Corpus['label']
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    cls=LogisticRegression(max_iter=300).fit(x_train,y_train)
    #sauver cls
    pickle.dump(cls,open("cls.pkl","wb"))

    return(cls.score(x_val,y_val))




