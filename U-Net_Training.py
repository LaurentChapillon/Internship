from __future__ import print_function
from tensorflow import keras
import configparser as ConfigParser
import os
import warnings
import os
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Activation, Permute, concatenate,Dropout,Conv2DTranspose,Lambda
from keras.models import Model
from keras.optimizers import Adam
from base_functions import get_train_data,get_test_data
from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
K.set_image_data_format('channels_last')

# --read configuration file-- #
config = ConfigParser.RawConfigParser()
config.read('./configuration.txt')

# --get parameters-- #
path_local = config.get('unet_parameters', 'path_local')
train_images_dir = path_local + config.get('unet_parameters', 'train_images_dir')
train_labels_dir = path_local + config.get('unet_parameters', 'train_labels_dir')
test_images_dir = path_local + config.get('unet_parameters', 'test_images_dir')
img_h = int(config.get('unet_parameters', 'img_h'))
img_w = int(config.get('unet_parameters', 'img_w'))
N_channels = int(config.get('unet_parameters', 'N_channels'))
C = int(config.get('unet_parameters', 'C'))

#if C > 2:
#    gt_list = eval(config.get('unet_parameters', 'gt_gray_value_list'))
#else:
gt_list = None

inputs = Input(shape=(img_h, img_w,N_channels))
# --Build a net work-- #
def get_net():
    #model=Sequential()

    #s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.5)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.5)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.5)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    """
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.5)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
 
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.5)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.5)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    """
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(p3)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.5)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.5)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.5)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(5, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    opt =keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

print('-' * 30)
print('Loading and pre-processing train data...')
print('-' * 30)

train_x, train_y = get_train_data(train_images_dir, train_labels_dir, img_h, img_w, C=C)
test_x = get_test_data(test_images_dir, img_h, img_w)
print('train_x size: ', np.shape(train_x))
print('train_y size: ', np.shape(train_y))

#assert(train_y.shape[1] == img_h and train_y.shape[2] == img_w)
#train_y = np.reshape(train_y, (train_y.shape[0], img_h*img_w, C))
model_path = path_local+config.get('unet_parameters', 'unet_model_dir')

# --Check whether the output path of the model exists or not-- #
#if os.path.isdir(model_path):
#    pass
#else:
#    os.mkdir(model_path)
# ------------------------------------ #
print('-' * 30)
print('Creating and compiling model...')
print('-' * 30)
model = get_net()
#model_checkpoint = ModelCheckpoint(model_path + '/unet.hdf5', monitor='loss', save_best_only=True)
"""
print('-' * 30)
print('Fitting model...')
print('-' * 30)
batch_size = int(config.get('unet_parameters', 'batch_size'))
epochs = int(config.get('unet_parameters', 'N_epochs'))
val_rate = config.get('unet_parameters', 'validation_rate')
hist = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False,
                 validation_split=float(val_rate), callbacks=[model_checkpoint], initial_epoch=0)
with open(model_path + '/unet.txt', 'w') as f:
    f.write(str(hist.history))
print('-' * 30)
print('Loading saved weights...')
print('-' * 30)
model.load_weights(model_path + '/unet.hdf5')
json_string = model.to_json()  # equal to: json_string = model.get_config()
open(model_path + '/unet.json', 'w').write(json_string)
model.save_weights(model_path + '/unet_weights.h5')
"""

checkpointer =ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        EarlyStopping(patience=2, monitor='val_loss'),
        TensorBoard(log_dir='logs')]
y_binary = to_categorical(train_y)

results = model.fit(train_x, y_binary, validation_split=0.3,class_weight='balanced', batch_size=30, epochs=200)
#model.save_weights(model_path + '/unet_weights.h5')

	


preds_train = model.predict(train_x[:int(train_x.shape[0]*0.9)], verbose=1)
preds_val = model.predict(train_x[int(train_x.shape[0]*0.9):], verbose=1)
preds_test = model.predict(test_x, verbose=1)
preds_test_t = (preds_test > 0.4).astype(np.uint8)
"""
class_index = np.argmax(preds_test[100,:,:,:], axis=2)
colors = {0 : [255, 0, 0], 1 : [0, 255, 0], 2 : [0, 0, 255], 3 : [127,127,127], 4 : [255,255,0]}
colored_image = np.array([colors[x] for x in np.nditer(class_index)], 
                         dtype=np.uint8)
output_image = np.reshape(colored_image, (64, 64, 3))
"""
ix = random.randint(0, len(preds_train))
plt.imshow(train_x[ix,:,:,0])
plt.figure(1)
plt.imshow(np.squeeze(train_y[ix,:,:,0]))
plt.figure(2)
plt.imshow(np.squeeze(preds_train[ix,:,:,0]))
plt.figure(3)

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_test))
#plt.imshow(train_x[int(train_x.shape[0]*0.9):][ix,:,:,0])
plt.figure(4)
plt.imshow(np.squeeze(train_y[int(train_y.shape[0]*0.9):][ix,:,:,0]))
plt.figure(5)
plt.imshow(np.squeeze(test_x[ix,:,:,0]))
plt.figure(6)
plt.imshow(np.squeeze(preds_test_t[ix,:,:,0]))

plt.imshow(np.squeeze(preds_test_t[ix,:,:,1]))
plt.show()