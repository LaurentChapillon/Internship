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
from sklearn.utils.fixes import loguniform
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

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

A=np.loadtxt('selected_variables_850-1800_methodDistMat.txt')
#A=A[351:730]

"""
my_data_test = my_data_test - my_data_test.mean()
my_data_test = my_data_test / np.abs(my_data_test).max()
my_data_train = my_data_train - my_data_train.mean()
my_data_train = my_data_train / np.abs(my_data_train).max()


scaler = StandardScaler()
scaler.fit(my_data_train)
my_data_train = scaler.transform(my_data_train)
my_data_test = scaler.transform(my_data_test)


  
my_data_train=my_data_train[:,1100:1575]
my_data_test=my_data_test[:,1100:1575]    
Loop=0
t = np.arange(0, 475, 1)
for i in range(0,np.size(A)):
        B=A[i]
        t=np.delete(t,B-Loop)
        Loop=Loop+1
Loop=0;
for i in range(0,np.size(t)):
        B=t[i]
        my_data_train = np.delete(my_data_train,(B-Loop),axis=1)
        my_data_test = np.delete(my_data_test,(B-Loop),axis=1)
        Loop=Loop+1
    
plt.figure(1)

t = np.arange(4000, 848, -2)
plt.plot(my_data_train[50,:])
"""
#mydata = (my_data_train * 255 / np.max(my_data_train)).astype('uint8')
#mydatatest = (my_data_test * 255 / np.max(my_data_test)).astype('uint8')
"""
parameters = param_grid = [
  {'C': [1, 10, 100, 1000,10000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000,10000], 'gamma': [0.1,0.01,0.001, 0.0001,0.00001], 'kernel': ['rbf']},
 ]
"""
max_abs_scaler = preprocessing.MaxAbsScaler()
my_data_train = max_abs_scaler.fit_transform(my_data_train)
my_data_test = max_abs_scaler.transform(my_data_test)
kernel = 1.0 * RBF(1.0)
clf = GaussianProcessClassifier(kernel=kernel,random_state=0)
#clf = MLPClassifier(alpha=1,max_iter=2000,verbose=True )
clf.fit(my_data_train, Label_train)
Svm_predict=clf.predict(my_data_test)
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
            
plt.figure(2)
plt.imshow(Test_RGB.astype(np.uint8))
