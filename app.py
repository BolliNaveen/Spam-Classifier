import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#lets load the saved vectorizer and naive model
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

#transform_text for fuction for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps=PorterStemmer()

def Transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    #removing special characters and retaining alphanumeric words
    text=[word for word in text if word.isalnum()]
    
    #removing stopwords and punctuation
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    #applying stemming
    text=[ps.stem(word) for word in text]
    
    return " ".join(text)

#steamlit code
#saving streamlit code
st.title('Navin"s Email SPAM Classifier')
input_sms=st.text_area('enter the message')

if st.button('predict'):
    #preprocessing
    transformed_sms=Transform_text(input_sms)
    #vectorize
    vector_input=tfidf.transform([transformed_sms])
    #predict
    result=model.predict(vector_input)[0]
    #display
    if result==1:
        st.header('SPAM')
    else:
        st.header('HAM')