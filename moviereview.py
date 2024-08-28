 import pandas as pd
 import numpy as np

data=pd.read_csv("/content/IMDB Dataset.csv")

data.head()

data.shape

type(data)

data.tail()

data["sentiment"].value_counts()

data.replace({"sentiment":{"positive":1,"negative":0}},inplace=True)

data.head()

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding,LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data,test_data=train_test_split(data,test_size=0.2,random_state=42)

train_data.shape

test_data.shape

tokenizer=Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])

X_train=pad_sequences(tokenizer.texts_to_sequences(train_data["review"]),maxlen=200)
X_test=pad_sequences(tokenizer.texts_to_sequences(train_data["review"]),maxlen=200)

Y_train=train_data["sentiment"]
Y_test=test_data["sentiment"]

model = Sequential()
model.add(Embedding(input_dim =5000, output_dim = 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = "sigmoid"))

model.summary()

model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, Y_train, epochs = 5, batch_size = 64, validation_split = 0.2)

model.save("model.h5")

import joblib
joblib.dump(tokenizer, "tokenizer.pkl")

def predictive_system(review):
  sequences = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequences, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment

from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("/content/model.h5")
tokenizer = joblib.load("/content/tokenizer.pkl")

def predictive_system(review):
  sequences = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequences, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment
     

review_sentiment = predictive_system("Beautiful cinematorgraphy")

review_sentiment

!pip install gradio

import gradio as gr
title = "MOVIE SENTIMENT ANALYSIS APPLICATION"

app = gr.Interface(fn = predictive_system, inputs="textbox", outputs="textbox", title=title)

app.launch(share=True)

