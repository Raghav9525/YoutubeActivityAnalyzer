from flask import Flask, render_template,request
import pickle
import numpy as np 

app = Flask(__name__)

model = pickle.load(open('set2_model.pkl','rb'))
keywords = pickle.load(open('keyword.pkl','rb'))
test_data= pickle.load(open('set2_test_data.pkl','rb'))


@app.route('/')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_movies',methods=['post'])
def recommend():
    index = int(request.form.get('user_input'))
    title = test_data[index] 

    x= model.predict(title.reshape(1,3000)).argmax(axis =1)

    if x==0:
        result= 'science&Technology'
    elif x==1:
        result= 'Food'
    elif x==2:
        result= 'Art&Music'
    else:
        result= 'Can,t predict, Try diffrent tiltle'
    return render_template('recommend.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

