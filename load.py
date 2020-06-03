import numpy as np 
import pandas as pd 
import streamlit as st
import cv2
import skimage.io as io 
import os 
import skimage.transform as trans 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
#import tensorflow.keras.backend as K

st.title('Coba loader')
@st.cache(allow_output_mutation=True)
def unet(pretrained_weights = None, input_size = (256,256, 1)):
    inputs = Input(input_size)
    #keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
    #                   padding='valid', data_format=None, 
    #                   dilation_rate=(1, 1), activation=None, 
    #                   use_bias=True, kernel_initializer='glorot_uniform', 
    #                   bias_initializer='zeros', kernel_regularizer=None, 
    #                   bias_regularizer=None, activity_regularizer=None, 
    #                   kernel_constraint=None, bias_constraint=None)
    
    # Rangkaian Extraction 1
    conv1 = Conv2D(64, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2))(conv1)

    # Rangkaian Extraction 2
    conv2 = Conv2D(128, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size = (2,2))(conv2)

    # Rangkaian Extraction 3
    conv3 = Conv2D(256, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size = (2,2))(conv3)

    # Rangkaian Extraction 4
    conv4 = Conv2D(512, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size = (2,2))(drop4)

    # Rangkaian Extraction 5
    conv5 = Conv2D(1024, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Rangkaian Expansion 1
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4, up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    # Rangkaian Expansion 2
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    # Rangkaian Expansion 3
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    # Rangkaian Expansion 4
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    # Rangkaian Expansion 5
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Membuat Model 
    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def saveResult(save_path, npyfile, flag_multi_class = False, num_class = 1):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:,:,0]
        print(img.shape)
        #img = trans.resize(img, (432,532)) # Gambar USG TA
        img = trans.resize(img, (512,470)) # Gambar USG Phantom
        io.imsave(os.path.join(save_path,"%d_predict.png"%i), img, check_contrast=False)

def testGenerator(test_path, num_image = 500, target_size = (256,256), flag_multi_class = False, as_gray = True):
    for i in range(num_image):
        #img = io.imread(os.path.join(test_path, "%d.png"%i), as_gray = as_gray)
        img = io.imread(os.path.join(test_path, "%d.png"%i), as_gray = as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)
        yield img

if __name__ == '__main__':
    st.title('My first app')
    #sentence = st.text_input('Input your sentence here:')
    model_checkpoint = ModelCheckpoint('unet_weights100.hdf5', monitor='loss',verbose=1, save_best_only=True)
    st.write('haloo')
    model = unet(pretrained_weights='unet_weights100.hdf5')
    run_button = st.button('Run', key = 0)
    if run_button:
        testGene = testGenerator('data/test-phantom') #Data Phantom
        results = model.predict_generator(testGene,167,verbose=1, callbacks = [model_checkpoint])#cv2.normalize(src= model.predict_generator(testGene,237,verbose=1, callbacks = [model_checkpoint]), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #saveResult('data/test', results) #Data GE
        saveResult('data/test-phantom', results) #Data Phantom
        st.success('Proses prediksi sukses, silahkan buka folder!')
        #if sentence:
        #    y_hat = model.predict(sentence)