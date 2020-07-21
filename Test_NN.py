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
from keras.models import Sequential
from keras.layers import Dense, Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
from keras.utils import np_utils 

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

my_data_train=np.reshape(my_data_train, [64,64,1626])
my_data_train=my_data_train.transpose(2,1,0)

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
my_data_train1=np.reshape(my_data_train1, [64,64,1626])
my_data_train1=my_data_train1.transpose(2,1,0)
Label_train = np.append (Label_train, Label_train1)
my_data_train = np.append(my_data_train, my_data_train1, axis = 0)

file_path_Csv_train=filedialog.askopenfilename()
file_path_Label_train=filedialog.askopenfilename()
my_data_train2 = genfromtxt(file_path_Csv_train, delimiter=';')
my_data_train2 = np.delete(my_data_train2, (0), axis=0)
my_data_train2 = np.delete(my_data_train2,(0), axis=1)
my_data_train2 = np.delete(my_data_train2,(0), axis=1)
my_data_train2 = np.delete(my_data_train2,(0), axis=1)
Label_train2 = image.imread(file_path_Label_train)
Label_train2 = np.asarray(Label_train2).reshape(-1)
Label_train2 = (Label_train2 * 255).astype(np.uint8)
print(np.amin(Label_train2))
print(np.amax(Label_train2))
my_data_train2=np.reshape(my_data_train2, [64,64,1626])
my_data_train2=my_data_train2.transpose(2,1,0)
Label_train = np.append (Label_train, Label_train2)
my_data_train = np.append(my_data_train, my_data_train2, axis = 0)


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
my_data_test=np.reshape(my_data_test, [64,64,1626])
my_data_test=my_data_test.transpose(2,1,0)
"""
A=np.loadtxt('selected_variables.txt')

Loop=0;
for i in range(0,np.size(A)):
    if  A[i]!=i:
        my_data_train = np.delete(my_data_train, (i-Loop), axis=1)
        my_data_test = np.delete(my_data_test, (i-Loop),axis=1)
        Loop=Loop+1
"""
my_data_test = my_data_test - my_data_test.mean()
my_data_test = my_data_test / np.abs(my_data_test).max()
my_data_train = my_data_train - my_data_train.mean()
my_data_train = my_data_train / np.abs(my_data_train).max()

my_data_train_train = my_data_train.reshape(my_data_train.shape[0], 1, 64, 64)
my_data_test = my_data_test.reshape(my_data_test.shape[0], 1, 64, 64)
Label_train = np_utils.to_categorical(Label_train, 5)
Label_test = np_utils.to_categorical(Label_test, 5)

model = Sequential()
 
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,64,64),data_format='channels_first'))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(my_data_train_train, Label_train, batch_size=32, nb_epoch=10, verbose=1)
clf=np.round(model.predict(my_data_test))

error = np.mean( clf != Label_test )

X_co=confusion_matrix(Label_test, clf)

clf=np.resize(clf,(64,64))
Test_RGB = np.zeros((64, 64, 3))
for i in range (0,64) :
    for j in range (0,64) :
        Value=clf[i,j]
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



