# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import optimizers
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
dataset=csv.reader(open('train.csv','r'))

#print dataset
X_train= np.zeros((2000,14))
Y_train= np.zeros((2000,1))
j=0
for data in dataset:
   if j>=1: 
    temp= data[1]
    #print (temp)
    Y_train[j-1]=data[2]
    i=0
    for element in temp:
        #print (element)
        if element=='A':
            X_train[j-1][i]=-1
            i=i+1
        if element=='C':
            X_train[j-1][i]=-0.5
            i=i+1
        if element=='G':
            X_train[j-1][i]=0.5
            i=i+1
        if element=='T':
            X_train[j-1][i]=1
            i=i+1
   j=j+1        
# ACGT
x= np.zeros((2000,14,1))
y= np.zeros((2000,1))
indices = np.arange(2000)# indices = the number of images in the source data set  
np.random.shuffle(indices)  
for i in indices:  
            x[i,:,0] = X_train[i,:]  
            y[i] =Y_train[i]  
         
#k=0
#while k<1000000:
# l=random.randint(0, 1999)
# m=random.randint(0, 1999)
# if l !=m:
 #    Y_train[l], Y_train[m]=Y_train[m], Y_train[l]
  #   X_train[l,:], X_train[m,:]=X_train[m,:], X_train[l,:]
 #k=k+1
#dataset = numpy.loadtxt("train.csv", delimiter=",")
# reshape to be [samples][pixels][width][height]
#rmsprop=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#sgd = SGD(lr=0.15, decay=1e-6, momentum=0, nesterov=True)
def SimpleMLP_model():
	# create model
    model = Sequential()
    model.add(LSTM(2,  input_shape=(14,1))) 
    model.add(Activation('relu'))
    model.add(Dropout(0.35))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = SimpleMLP_model()
M=1800
history=model.fit(x[0:M,:,:], y[0:M],validation_data=(x[M+1:1999,:,:],  y[M+1:1999]), nb_epoch=1000, batch_size=50)
#history=model.fit(X_train[0:M,:], Y_train[0:M],validation_data=(X_train[M+1:1999,:],  Y_train[M+1:1999]), nb_epoch=50, batch_size=1)
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))
model.save('gene.h5') 

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'], loc='upper right')
plt.title('binary_crossentropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train','validation'], loc='lower right')
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()