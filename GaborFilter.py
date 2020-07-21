import numpy as np
import cv2
from tkinter import filedialog
from matplotlib import image
from matplotlib import pyplot as plt
from numpy import genfromtxt
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd


df=pd.DataFrame()
file_path_Csv_train=filedialog.askopenfilename()
my_data_train = genfromtxt(file_path_Csv_train, delimiter=';')
#my_data_train = (my_data_train * 255 / np.max(my_data_train)).astype('uint8')

num=0
ksize=7
sigma=1.0
gamma=0.5
for i in range (0,316):
    img=my_data_train[:,i]
    img=np.reshape(img,[64,64])
#    for theta in np.arange(np.pi/4, np.pi, np.pi / 4):
#        for sigma in (3,5):
#            for gamma in (0.05,0.5):
#                for lambd in np.arange(np.pi/4,np.pi,np.pi/4):
    for theta in np.arange(np.pi/8, np.pi, np.pi / 8):
        for lambd in np.arange(np.pi/4, np.pi, np.pi/4): 
            kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma,0,ktype=cv2.CV_64F)
            #◘kernel /= 1.5*kernel.sum()
            gabor_label='Gabor'+str(num)
            res1=cv2.filter2D(img,cv2.CV_64F,kernel)
            filtered_img=res1.reshape(-1)
            df[gabor_label] =filtered_img
            num+=1




          

df.to_csv('Gabor_4.csv',sep=';')

