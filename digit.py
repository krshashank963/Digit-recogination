# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:10:10 2020

@author: SHASHANK RAJPUT
"""

from keras.models import Sequential#initialize the CNN
from keras.layers import Dense
from keras.layers import Convolution2D#for convolution layer
from keras.layers import MaxPooling2D#poolin step
from keras.layers import Flatten
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
print(mnist)
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# plt.imshow(x_train[3],cmap=plt.cm.hot)
# plt.show()

#normalizxee the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#print the normalize equivalent
print(x_train)
plt.imshow(x_train[3],cmap=plt.cm.binary)
plt.show()

model = Sequential()

#to make a vector of iage specification
model.add(Flatten())
#hiddden layer
model.add(Dense(128, activation='relu'))
#2nd hidden layer
model.add(Dense(128, activation='relu'))

#output layer
model.add(Dense(10, activation='sigmoid'))

#compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics =['accuracy'])

#fit the model
model.fit(x_train, y_train, epochs=4)

#makeing the predication
prediction = model.predict(x_test)

print(prediction)

result = np.argmax(prediction[7])

print(x_train)
plt.imshow(x_test[7],cmap=plt.cm.binary)
plt.show()
print(result)




