from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask, render_template,request
import pickle
import numpy as np 
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
ps=PorterStemmer()



model = pickle.load(open('set2_model.pkl','rb'))
keywords = pickle.load(open('keyword.pkl','rb'))
remove_special1 = pickle.load(open('remove_special.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_movies',methods=['post'])
def recommend():
    title = str(request.form.get('user_input')).lower()
    #remove special char
    title=remove_special1(title)
    # remove stop words
    title= remove_stopWords1(title)

    #steming 
    title= stem_word1(title)

    title_list =" ".join(title)

    # title = list(str(request.form.get('user_input')).split(" "))
    # title_list =" ".join(title)

    # converting title into list of words
    # title_list=[]
    # for i in title.split(" "):
    #     title_list.append(i)
    title_list = [title_list]

    vectorize = CountVectorizer(vocabulary=keywords)
    x_test =vectorize.transform(title_list)


    x= model.predict(x_test.reshape(1,3200)).argmax(axis =1)

    if x==0:
        result= 'computer science'
    elif x==1:
        result= 'Food'
    elif x==2:
        result= 'Art&Music'
    else:
        result= 'Can,t predict, Try diffrent tiltle'
    return render_template('recommend.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

