import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from models.EEGNet import *

from preprocessing import *

import joblib

import serial

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='COM51',
    #'/dev/tty.usbserial-10',
    baudrate=9600,
    timeout=.1
)
ser.isOpen()

##################### SCALER #########################################
scaler_teeth = joblib.load(r'.\checkpoints\teeth\scaler.save') 
scaler_eyebrows = joblib.load(r'.\checkpoints\eyebrows\scaler.save') 
######################################################################

n_timesteps = 128

#############################################################################
####################### MODELING ############################################
#############################################################################
base_model = EEGNet_SSVEP(
    nb_classes = 1, Chans = 8, Samples = 128, 
    dropoutRate = 0.5, kernLength = 100, F1 = 32, 
    D = 1, F2 = 32, dropoutType = 'Dropout'
)
x = base_model.layers[-3].output
x = Dense(128*3, activation='relu')(x)
x = Reshape((128, 3))(x)
x = Activation('softmax', name = 'softmax')(x)
model = Model(inputs=base_model.input, outputs=x)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)

# model.load_weights(r'.\checkpoints\teeth\eegnet_v1_tp9_tp10.weights.h5')
model.load_weights(r'.\checkpoints\eegnet_eyebrows_teeth.weights.h5')

#############################################################################
####################### END MODELING ########################################
#############################################################################


""" 1. CONNECT TO EEG STREAM """
# Search for active LSL streams
print('Looking for an EEG stream...')
streams = resolve_byprop('type', 'EEG', timeout=2)
if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')
print("Start acquiring data")
inlet = StreamInlet(streams[0], max_chunklen=128)
eeg_time_correction = inlet.time_correction()

while True:
    # inlet.pull_chunk(timeout=1.0, max_samples=128)
    eeg_data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=n_timesteps)
    eeg_data = np.array(eeg_data)[:, :-1]  # sample, channel

    ########################### Preprocessing ###################################
    eeg_data = eeg_data.transpose((1, 0))
    teeth_feature = eeg_data.copy()
    eyebrows_feature = eeg_data.copy()

    for column in range(eeg_data.shape[0]):
        teeth_feature[column] = filter_teeth(eeg_data[column])
        eyebrows_feature[column] = filter_eyebrows(eeg_data[column])

    teeth_feature = scaler_teeth.transform(teeth_feature.T).T
    eyebrows_feature = scaler_eyebrows.transform(eyebrows_feature.T).T
    
    input = np.concatenate([teeth_feature, eyebrows_feature], axis=0)
    input = np.expand_dims(input, 0)
    input = np.expand_dims(input, -1)
    assert input.shape == (1, 8, 128, 1)
    #############################################################################


    ############################### Inference ###################################
    y_pred = model.predict(input)
    y_pred = np.argmax(y_pred, 2)[0]
    #############################################################################
    print(y_pred)

    pred = np.max(y_pred)
    print(pred)

    if pred == 1:
        print('Left')
        ser.write(bytes('3', 'ascii'))
    elif pred == 2:
        print('Right')
        ser.write(bytes('4', 'ascii'))
    elif pred == 0:
        print('Stop')
        ser.write(bytes('0', 'ascii'))
