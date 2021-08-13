import streamlit as st 
import pickle
import urllib
import gensim.downloader as api
import numpy as np


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
            #logging.error('Word is not found in the trained vector dataset')
            pass
    features = np.array(features)
    print(count)
    features = np.divide(features,count)
    return features

def vector(text,wv):   ### returns vector representation of all news articles
    #vec = [avg(wv,x) for x in df['Text']]
    text = ' '.join(text)
    vec = avg(wv,text)
    return np.array(vec)


def preprocess(text):

    k = text.split('.')
    j = [i.lower() for x in k for i in x]

    return j 

def main():

    st.title('News classifier app')
    activities=['NLP', 'Prediction']
    choice = st.sidebar.selectbox('Choose activity',activities)
    with open('model_registry/mod','rb') as f:
        mod = pickle.load(f)
    if choice=='Prediction':
        st.info('Prediction With Ml')
        text = st.text_area('Enter Text')

        text = preprocess(text)
        wv = word()
        vec = vector(text,wv)
        vec = vec.reshape(-1,300)
        pred = mod.predict(vec)
        print(pred)
        st.write(pred)
        st.success(pred[0])


        #url = st.text_input('The url link')

    


if __name__ =='__main__':

    main()