from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask, render_template,request
import pickle
import numpy as np 
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
ps=PorterStemmer()

goal=""

model = pickle.load(open('D:\Flask\SET2\set2_model.pkl','rb'))
keywords = pickle.load(open('D:\Flask\SET2\keyword.pkl','rb'))


app = Flask(__name__)

def remove_special(title):
    f_text=''
    for i in title.split(' '): 
        for j in i:
            if j.isalpha():
               f_text=f_text+j 
        f_text=f_text+' '
    return f_text

#remoce stopword
def remove_stopWords(f_text):
    stop_w = set(stopwords.words('english'))
    x=[]
    for i in f_text.split():
        if i not in stop_w:
            x.append(i)
    f_sentance=x[:]
    x.clear()
    return f_sentance

# steming
def stem_word(text):
    y=[]
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z

@app.route('/')
def homepage():
    goal = str(request.form.get('goal')).lower()
    return render_template('home.html',methods=['post'])

@app.route('/classify' ,methods=['GET', 'POST'])
def classify():
    return render_template('recommend.html')



@app.route('/recommend_movies',methods=['post'])
def recommend():
    title = str(request.form.get('user_input')).lower()
    #remove special char
    title=remove_special(title)
    # remove stop words
    title= remove_stopWords(title)

    #steming 
    title= stem_word(title)

    title_list =" ".join(title)
    title_list = [title_list]

    vectorize = CountVectorizer(vocabulary=keywords)
    x_test =vectorize.transform(title_list)


    x= model.predict(x_test.reshape(1,3000)).argmax(axis =1)

    if x==0:
        result= 'computer science'
    elif x==1:
        result= 'Food'
    elif x==2:
        result= 'Art&Music'
    else:
        result= 'Category is different from computerScience, Food, Art&Music, Please Try diffrent title'

    if(goal != result):
        msg= "you are getting away from your goal"
    else:
        msg= "You are focused"

    return render_template('recommend.html',result= result , msg=msg,goal= goal)

if __name__ == '__main__':
    app.run(debug=True)

