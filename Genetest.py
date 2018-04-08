# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy
import numpy as np
import csv
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
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
dataset=csv.reader(open('test.csv','r'))
from keras.models import load_model  
my_model = load_model('gene.h5') 
#print dataset
X_test= np.zeros((400,14))
Y_test= np.zeros((400,1))
j=0
for data in dataset:
   if j>=1: 
    temp= data[1]
    #print (temp)
    i=0
    for element in temp:
        #print (element)
        if element=='A':
            X_test[j-1][i]=0.25
            i=i+1
        if element=='C':
            X_test[j-1][i]=0.5
            i=i+1
        if element=='G':
            X_test[j-1][i]=0.75
            i=i+1
        if element=='T':
            X_test[j-1][i]=1
            i=i+1
   j=j+1        
# ACGT
k=0

Y_test = my_model.predict(X_test[:,:])  
print('Label of testing sample', np.argmax(Y_test))  
csvFile2 = open('ts.csv','w', newline='') 
writer = csv.writer(csvFile2)
writer.writerow(['id,prediction'])
for i in range(400):
    if Y_test[i]<0.5:
        writer.writerow([i, 0])
    if Y_test[i]>=0.5:
        writer.writerow([i,1])
csvFile2.close()