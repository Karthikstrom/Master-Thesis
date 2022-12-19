# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:33:12 2022

@author: Karthikeyan
"""
#%% Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
import seabon as sns

#%% Importing dataset

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

#%% Visualizing the data

plt.matshow(X_train[800])

#%% Scaling of data
X_train=X_train/255
X_test=X_test/255
#%% Flatten to fit the pixels in the input layer format
#using reshape

X_train_flattened=X_train.reshape(len(X_train),28*28)
X_test_flattened=X_test.reshape(len(X_test),28*28)

#%% Creating a simple NN

model=keras.Sequential([
# keras.layers.Flatten(input_shape=(28,28))
    keras.layers.Dense(100, input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
    ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.fit(X_train_flattened,y_train,epochs=5)

#%% Evaluation on the test dataset
model.evaluate(X_test_flattened,y_test) 

#%% Sample prediction

y_predicted=model.predict(X_test_flattened)
y_predicted[0]

# so the o/p of the prediction would be the output layer 
# activation values so finding the highest one would give us the
# resulting prediction

np.argmax(y_predicted[1])

#%% Converting all the predictions into labels

y_predicted_labels= [np.argmax(i) for i in y_predicted]

#%% Building a confusion matrix

cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
