# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:00:30 2020

@author: Laurent CHAPILLON
"""
import numpy as np
from tkinter import filedialog
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib import image
from matplotlib import pyplot as plt
from matplotlib import cm
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix

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

A=np.loadtxt('selected_variables.txt')


Loop=0;
for i in range(0,np.size(A)):
    if  A[i]==i:
        my_data_train = np.delete(my_data_train, (i-Loop), axis=1)
        my_data_test = np.delete(my_data_test, (i-Loop),axis=1)
        Loop=Loop+1
my_data_test = my_data_test - my_data_test.mean()
my_data_test = my_data_test / np.abs(my_data_test).max()
my_data_train = my_data_train - my_data_train.mean()
my_data_train = my_data_train / np.abs(my_data_train).max()

lda = LinearDiscriminantAnalysis(solver='eigen')
X_lda=lda.fit(my_data_train, Label_train)
X_lda_test = lda.predict(my_data_test)
X_scaling=lda.scalings_
X_co=confusion_matrix(Label_test, X_lda_test)
Accuracy1=lda.score(my_data_test,Label_test)

X_lda_test=np.resize(X_lda_test,(64,64))
Test_RGB = np.zeros((64, 64, 3))
for i in range (0,64) :
    for j in range (0,64) :
        Value=X_lda_test[i,j]
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






