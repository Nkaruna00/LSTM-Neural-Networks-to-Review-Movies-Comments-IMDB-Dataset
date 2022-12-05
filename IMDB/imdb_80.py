#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 01:38:56 2019

@author: nithushan
"""

# Import libraries
import numpy as np


np_load_old = np.load

np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# Load the IMDB Dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)


# Pad the sequence to have same length of sentences
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)


# Build the LSTM Model
model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


# COmpile the model with Adam optimizers
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


#Train the model 
model.fit(x_train, y_train,batch_size=32,epochs=4,validation_data=(x_test, y_test))

# Evaluate the model
scores = model.evaluate(x_test, y_test,batch_size=32,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save("imdb.h5")
