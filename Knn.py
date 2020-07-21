# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:40:52 2020

@author: admin
"""


import numpy as np
from tkinter import filedialog
from matplotlib import image
from matplotlib import pyplot as plt
from numpy import genfromtxt
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import loguniform
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

Lumen_RGB=[150,20,150]
Fibers_RGB=[175,50,255]
Vessels_RGB=[227,127,100]
Rays_RGB=[50,50,100]
G_layers_RGB=[20,255,255]

file_path_Csv_train=filedialog.askopenfilename()
file_path_Label_train=filedialog.askopenfilename()
my_data_train = genfromtxt(file_path_Csv_train, delimiter=';')
Label_train = image.imread(file_path_Label_train)
Label_train = np.asarray(Label_train).reshape(-1)
Label_train = (Label_train * 255).astype(np.uint8)
print(np.amin(Label_train))
print(np.amax(Label_train))


    
file_path_Csv_train=filedialog.askopenfilename()
file_path_Label_train=filedialog.askopenfilename()
my_data_train1 = genfromtxt(file_path_Csv_train, delimiter=';')
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
Label_test = image.imread(file_path_Label_test)
Label_test = np.asarray(Label_test).reshape(-1)
Label_test = (Label_test * 255).astype(np.uint8)
print(np.amin(Label_test))
print(np.amax(Label_test))
print(np.amin(Label_train))

max_abs_scaler = preprocessing.MaxAbsScaler()
my_data_train = max_abs_scaler.fit_transform(my_data_train)
my_data_test = max_abs_scaler.transform(my_data_test)

clf = neighbors.KNeighborsClassifier(n_neighbors=7)
clf.fit(my_data_train, Label_train)
Knn_predict=clf.predict(my_data_test)
error = np.mean( Knn_predict != Label_test )
X_co=confusion_matrix(Label_test, Knn_predict)
Knn_predict=np.resize(Knn_predict,(64,64))
Test_RGB = np.zeros((64, 64, 3))
for i in range (0,64) :
    for j in range (0,64) :
        Value=Knn_predict[i,j]
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
            

plt.imshow(Test_RGB.astype(np.uint8))
