import pandas as pd
from sklearn.model_selection import train_test_split
import gensim.downloader as api
import numpy as np
import gensim.downloader as api
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
import os
from config import filename 
#os.mkdir('model_registry')

def readfile(filename):    ##Reads File and returns dataframe
    df = pd.read_csv(filename)
    return df

def word():   ## Returns the word2vec pretrained model
    wv = api.load('word2vec-google-news-300')
    return wv

def avg(wv,word):  ###computes the avg vector representation for each news article
    features = np.zeros(300,dtype = 'float64')
    word = word.split(' ')
    count = 0
    for i in word:
        try:
            count+=1
            features = features+wv[i]
        except:
            pass
    features = np.array(features)
    print(count)
    features = np.divide(features,count)
    return features

def vector(df,wv):   ### returns vector representation of all news articles
    vec = [avg(wv,x) for x in df['Text']]
    return np.array(vec)
    

def splitdatast(df,vec): ### splits the dataset into train test 
    x_train,x_test,y_train,y_test = train_test_split(vec,df['Category'],test_size=0.2)
    return x_train,x_test,y_train,y_test

def model_building(x_train,y_train):  ###Trains the model and stores it in the model registry
    param = {
    'bootstrap': [True],
    'max_depth': [1,5],
    'max_features':[2,3],
    'min_samples_leaf':[3,5],
    'min_samples_split':[4],
    'n_estimators':[5]
    }
    mod = RandomForestClassifier()
    grid = GridSearchCV(estimator=mod,param_grid=param,cv=3,n_jobs=-1,verbose=2)
    grid.fit(x_train,y_train)
    mod1 =grid.best_estimator_
    with open('model_registry/mod','wb') as f:
        pickle.dump(mod1,f)
    y_pred=mod1.predict(x_test)
    print(accuracy_score(y_pred,y_test))
    






if __name__=='__main__':

    df =readfile(filename)
    wv = word()
    vec = vector(df,wv)

    x_train,x_test,y_train,y_test = splitdatast(df,vec)
    model_building(x_train,y_train)









