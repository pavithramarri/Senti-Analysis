from flask import Flask, render_template, request
import pickle


filename = 'restaurant-sentiment-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
	return render_template('review.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = message.split('\n')    
    vect = cv.transform(data).toarray()
    my_prediction = classifier.predict(vect)
    d = {'p':0,'n':0}
    for i in my_prediction:
        if i==0:
            d['n']+=1
        else:
            d['p']+=1    
    return render_template('result.html', prediction=my_prediction, p=d['p'],n=d['n'])
    	    

if __name__ == '__main__':
	app.run(debug=True)