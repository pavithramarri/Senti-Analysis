import re  
import nltk
#nltk.download('wordnet')

def preprocessor(text):
  emotions=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)(...)',text)
  text = re.sub('[\W]+',' ',text.lower()) + ' '.join(emotions).replace('-','')
  return text
  
import numpy as np
import pandas as pd
import sklearn
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pickle


#IMPORTING DATA

df = pd.read_csv('C:\Users\PAVITHRA\Documents\Python Scripts\pavi\datasets_6660_9643_Restaurant_Reviews (1).tsv', delimiter = '\t')

df['Review']=df['Review'].apply(preprocessor)


def tokenizer(text):
  return text.split()

df['Review']=df['Review'].apply(tokenizer)

porter = SnowballStemmer('english',ignore_stopwords=False)

def stem_it(text):
  return [porter.stem(word) for word in text]

df['Review']=df['Review'].apply(stem_it)

lemmatizer = WordNetLemmatizer()

def lemmit(text):
  return [lemmatizer.lemmatize(word,pos='a') for word in text]

df['Review']=df['Review'].apply(lemmit)

stop_word = set(['a','t','d','y'])

def stop(text):
  review  = [word for word in text if not word in stop_word]
  return review

df['Review']=df['Review'].apply(stop)

df['Review']=df['Review'].apply(' '.join)

from sklearn.feature_extraction.text import TfidfVectorizer


tfidf=TfidfVectorizer(strip_accents=None, lowercase=False,use_idf=True,norm='l2',smooth_idf=True)

y=df.Liked.values
x=tfidf.fit_transform(df['Review'])

# Creating a pickle file for the CountVectorizer
pickle.dump(tfidf, open('cv-transform.pkl', 'wb'))

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)

from sklearn.linear_model import LogisticRegressionCV

clf=LogisticRegressionCV(random_state=0,verbose=3,
                         max_iter=300)
clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score

predictions=clf.predict(x_test)

acc_log = accuracy_score(predictions, y_test)*100
print(acc_log)

filename = 'restaurant-sentiment-model.pkl'
pickle.dump(clf, open(filename, 'wb'))