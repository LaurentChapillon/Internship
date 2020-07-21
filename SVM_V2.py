# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:52:58 2020

@author: admin
"""
import numpy as np
from tkinter import filedialog
from matplotlib import image
from matplotlib import pyplot as plt
from numpy import genfromtxt
from sklearn import svm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import scipy.signal
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from numpy import load

Lumen_RGB=[150,20,150]
Fibers_RGB=[175,50,255]
Vessels_RGB=[227,127,100]
Rays_RGB=[50,50,100]
G_layers_RGB=[20,255,255]

file_path_Csv_train=filedialog.askopenfilename()
file_path_Label_train=filedialog.askopenfilename()
my_data_train = genfromtxt(file_path_Csv_train, delimiter=';')
my_data_train = np.delete(my_data_train, (0), axis=0)
my_data_train = np.delete(my_data_train,(0), axis=1)
my_data_train = np.delete(my_data_train,(0), axis=1)
my_data_train = np.delete(my_data_train,(0), axis=1)
Label_train = image.imread(file_path_Label_train)
Label_train = np.asarray(Label_train).reshape(-1)
Label_train = (Label_train * 255).astype(np.uint8)
print(np.amin(Label_train))
print(np.amax(Label_train))

for i in range (0,4096):
    X=my_data_train[i,:]
    yhat = scipy.signal.savgol_filter(X, 5, 3)
    my_data_train[i,:]=yhat
    
file_path_Csv_train=filedialog.askopenfilename()
file_path_Label_train=filedialog.askopenfilename()
my_data_train1 = genfromtxt(file_path_Csv_train, delimiter=';')
my_data_train1 = np.delete(my_data_train1, (0), axis=0)
my_data_train1 = np.delete(my_data_train1,(0), axis=1)
my_data_train1 = np.delete(my_data_train1,(0), axis=1)
my_data_train1 = np.delete(my_data_train1,(0), axis=1)
Label_train1 = image.imread(file_path_Label_train)
Label_train1 = np.asarray(Label_train1).reshape(-1)
Label_train1 = (Label_train1 * 255).astype(np.uint8)
print(np.amin(Label_train1))
print(np.amax(Label_train1))

for i in range (0,4096):
    X=my_data_train1[i,:]
    yhat = scipy.signal.savgol_filter(X, 5, 3)
    my_data_train1[i,:]=yhat


Label_train = np.append (Label_train, Label_train1)
my_data_train = np.append(my_data_train, my_data_train1, axis = 0)

file_path_Csv_train=filedialog.askopenfilename()
file_path_Label_train=filedialog.askopenfilename()
my_data_train1 = genfromtxt(file_path_Csv_train, delimiter=';')
my_data_train1 = np.delete(my_data_train1, (0), axis=0)
my_data_train1 = np.delete(my_data_train1,(0), axis=1)
my_data_train1 = np.delete(my_data_train1,(0), axis=1)
my_data_train1 = np.delete(my_data_train1,(0), axis=1)
Label_train1 = image.imread(file_path_Label_train)
Label_train1 = np.asarray(Label_train1).reshape(-1)
Label_train1 = (Label_train1 * 255).astype(np.uint8)
print(np.amin(Label_train1))
print(np.amax(Label_train1))

for i in range (0,4096):
    X=my_data_train1[i,:]
    yhat = scipy.signal.savgol_filter(X, 5, 3)
    my_data_train1[i,:]=yhat

Label_train = np.append (Label_train, Label_train1)
my_data_train = np.append(my_data_train, my_data_train1, axis = 0)


file_path_Csv_test=filedialog.askopenfilename()
file_path_Label_test=filedialog.askopenfilename()
my_data_test = genfromtxt(file_path_Csv_test, delimiter=';')
my_data_test = np.delete(my_data_test, (0), axis=0)
my_data_test = np.delete(my_data_test,(0), axis=1)
my_data_test = np.delete(my_data_test,(0), axis=1)
my_data_test = np.delete(my_data_test,(0), axis=1)
Label_test = image.imread(file_path_Label_test)
Label_test = np.asarray(Label_test).reshape(-1)
Label_test = (Label_test * 255).astype(np.uint8)
print(np.amin(Label_test))
print(np.amax(Label_test))
print(np.amin(Label_train))

file_path_Csv_test=filedialog.askopenfilename()
my_data_train_gabor = load(file_path_Csv_test)
file_path_Csv_test=filedialog.askopenfilename()
my_data_train_gabor1 = load(file_path_Csv_test)
my_data_train_gabor = np.append(my_data_train_gabor, my_data_train_gabor1, axis = 0)
file_path_Csv_test=filedialog.askopenfilename()
my_data_train_gabor1 = load(file_path_Csv_test)
my_data_train_gabor = np.append(my_data_train_gabor, my_data_train_gabor1, axis = 0)

file_path_Csv_test=filedialog.askopenfilename()
my_data_test_gabor = load(file_path_Csv_test)



for i in range (0,4096):
    X=my_data_test[i,:]
    yhat = scipy.signal.savgol_filter(X, 5, 3)
    my_data_test[i,:]=yhat

A=np.loadtxt('selected_variables.txt')

Loop=0;
for i in range(0,np.size(A)):
    if  A[i]!=i:
        my_data_train = np.delete(my_data_train, (i-Loop), axis=1)
        my_data_test = np.delete(my_data_test, (i-Loop),axis=1)
        my_data_test_gabor=np.delete(my_data_test_gabor,(i-Loop),axis=1)
        my_data_train_gabor=np.delete(my_data_train_gabor,(i-Loop),axis=1)
        Loop=Loop+1

"""
my_data_test = my_data_test - my_data_test.mean()
my_data_test = my_data_test / np.abs(my_data_test).max()
my_data_train = my_data_train - my_data_train.mean()
my_data_train = my_data_train / np.abs(my_data_train).max()
"""
"""
scaler = StandardScaler()
scaler.fit(my_data_train)
my_data_train = scaler.transform(my_data_train)
my_data_test = scaler.transform(my_data_test)
"""
my_data_train_gabor = my_data_train_gabor - my_data_train_gabor.mean()
my_data_train_gabor = my_data_train_gabor / np.abs(my_data_train_gabor).max()
my_data_test_gabor = my_data_test_gabor - my_data_test_gabor.mean()
my_data_test_gabor = my_data_test_gabor / np.abs(my_data_test_gabor).max()
#mydata = (my_data_train * 255 / np.max(my_data_train)).astype('uint8')
#mydatatest = (my_data_test * 255 / np.max(my_data_test)).astype('uint8')
clf =svm.SVC(kernel='rbf',class_weight='balanced',C=10,gamma=0.01)
clf.fit(my_data_train_gabor, Label_train)
Svm_predict=clf.predict(my_data_test_gabor)
error = np.mean( Svm_predict != Label_test )

X_co=confusion_matrix(Label_test, Svm_predict)

Svm_predict=np.resize(Svm_predict,(64,64))
Test_RGB = np.zeros((64, 64, 3))
for i in range (0,64) :
    for j in range (0,64) :
        Value=Svm_predict[i,j]
        if Value ==1 :
            Test_RGB[i,j,0]=Lumen_RGB[0]
            Test_RGB[i,j,1]=Lumen_RGB[1]
            Test_RGB[i,j,2]=Lumen_RGB[2]
        if Value ==2 :
            Test_RGB[i,j,0]=Fibers_RGB[0]
            Test_RGB[i,j,1]=Fibers_RGB[1]
            Test_RGB[i,j,2]=Fibers_RGB[2]
        if Value ==3 :
            Test_RGB[i,j,0]=Vessels_RGB[0]
            Test_RGB[i,j,1]=Vessels_RGB[1]
            Test_RGB[i,j,2]=Vessels_RGB[2]
        if Value ==4 :
            Test_RGB[i,j,0]=Rays_RGB[0]
            Test_RGB[i,j,1]=Rays_RGB[1]
            Test_RGB[i,j,2]=Rays_RGB[2]
        if Value ==5 :
            Test_RGB[i,j,0]=G_layers_RGB[0]
            Test_RGB[i,j,1]=G_layers_RGB[1]
            Test_RGB[i,j,2]=G_layers_RGB[2]
            
plt.figure(1)
plt.imshow(Test_RGB.astype(np.uint8))