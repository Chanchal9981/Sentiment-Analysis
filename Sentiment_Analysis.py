# -*- coding: utf-8 -*-


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import keras
import pandas as pd
import numpy as np
import tensorflow

data=r"/content/drive/MyDrive/NLP_Data_Sets-20/NLP_Data_Sets/train.txt"
dataset=pd.read_csv(data,sep=";")
dataset

dataframe=dataset.rename(columns={"i didnt feel humiliated":"text","sadness":"Mood"})
dataframe

dataframe["Mood"].value_counts()

max_fatures=2000

tokenizer=Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(dataframe['text'].values)
X=tokenizer.texts_to_sequences(dataframe['text'].values)
X=pad_sequences(X, 28) 

Y=pd.get_dummies(dataframe['Mood']).values
print(X.shape)
print(Y.shape)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)
print(X_train.shape)
print(Y_train.shape)

embed_dim=128
lstm_out=196

model=Sequential()
model.add(Embedding(max_fatures,embed_dim,input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out,dropout=0.3,recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(128,recurrent_dropout=0.2))
model.add(Dense(6,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

batch_size=128

model.fit(X_train,Y_train,epochs=10,batch_size=batch_size,validation_data=(X_test,Y_test))

batch_size=128

model.fit(X_train,Y_train,epochs=10,batch_size=batch_size,validation_data=(X_test,Y_test))

batch_size=128

model.fit(X_train,Y_train,epochs=10,batch_size=batch_size,validation_data=(X_test,Y_test))

model.save("NLU_Assign.h5")

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
dataframe['selected_text1'] = dataframe['text'].apply(lambda x: sid.polarity_scores(x))
def convert(x):
 if x <= -0.05:
     return "negative"
 elif x >= 0.05:
   return "positive"
 else:
   return "positive"

dataframe['sentiment'] = dataframe['selected_text1'].apply(lambda x:convert(x['compound']))

dataframe

import re
def prop(text):
  words = re.findall('\w+', text.lower())
  uniq_words = set(words)
  return uniq_words
dataframe["selected_text"]=dataframe["text"].apply(prop)
dataframe.head()

data_frame=dataframe[["text","Mood","selected_text","sentiment"]]
data_frame.to_csv("train.csv")

dataframe.to_csv("train.csv")

valid_path=r"/content/drive/MyDrive/NLP_Data_Sets-20/NLP_Data_Sets/val.txt"
valid_read=pd.read_csv(valid_path,sep=";")
valid_dataframe=valid_read.rename(columns={"im feeling quite sad and sorry for myself but ill snap out of it soon":"text","sadness":"Mood"})
valid_dataframe.head()

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
valid_dataframe['selected_text'] = valid_dataframe['text'].apply(lambda x: sid.polarity_scores(x))
def convert(x):
 if x <= .5:
     return "negative"
 elif x >= .7:
   return "positive"
 else:
   return "positive"

valid_dataframe['sentiment'] = valid_dataframe['selected_text'].apply(lambda x:convert(x['compound']))

import re
def prop(text):
  words = re.findall('\w+', text.lower())
  uniq_words = set(words)
  return uniq_words
valid_dataframe["selected_text"]=valid_dataframe["text"].apply(prop)
valid_dataframe.to_csv("validation.csv")

test_data=r"/content/drive/MyDrive/NLP_Data_Sets-20/NLP_Data_Sets/test.txt"
test_read=pd.read_csv(test_data,sep=";")
test_dataframe=test_read.rename(columns={"im feeling rather rotten so im not very ambitious right now":"Text","sadness":"Mood"})
test_dataframe.head()

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
test_dataframe['selected_text'] = test_dataframe['Text'].apply(lambda x: sid.polarity_scores(x))
def convert(x):
 if x <= -0.05:
     return "negative"
 elif x >= 0.05:
   return "positive"
 else:
   return "positive"

test_dataframe['sentiment'] = test_dataframe['selected_text'].apply(lambda x:convert(x['compound']))

import re
def prop(text):
  words = re.findall('\w+', text.lower())
  uniq_words = set(words)
  return uniq_words
test_dataframe["selected_text"]=test_dataframe["Text"].apply(prop)
test_dataframe.head()
test_dataframe.to_csv("test.csv")

test_dataframe

