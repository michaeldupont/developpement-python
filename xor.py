# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:03:50 2018

@author: fmdu
"""

from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#Training examples
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0], [1], [1], [0]]

input_layer = input_data(shape=[None, 2]) #input layer of size 2
hidden_layer = fully_connected(input_layer , 2, activation='tanh') #hidden layer of size 2
output_layer = fully_connected(hidden_layer, 1, activation='tanh') #output layer of size 1

#use Stohastic Gradient Descent and Binary Crossentropy as loss function
regression = regression(output_layer , optimizer='sgd', loss='binary_crossentropy', learning_rate=5)
model = DNN(regression)

#fit the model
model.fit(X, Y, n_epoch=5000, show_metric=True);

#predict all examples
print ('Expected:  ', [i[0] > 0 for i in Y])
print ('Predicted: ', [i[0] > 0 for i in model.predict(X)])

model.get_weights(hidden_layer.W)
model.get_weights(output_layer.W)

model.save("tflearn-xor")
    
    
    
    