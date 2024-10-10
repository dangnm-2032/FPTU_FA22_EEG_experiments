import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import Model, load_model
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
from utils import *

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
import joblib
label_name = ['eyebrows', 'left', 'right', 'both', 'teeth']
scalers = {}
for label in label_name:
    scalers[label] = joblib.load(rf'.\pipeline_{label}\checkpoints\scaler.save')
######################################################################

n_timesteps = 128

####################### MODEL ############################################
model = load_model(r'.\checkpoints\orthogonal_false.keras')
####################### END MODEL ########################################

####### FILTER #############################################
filters = {
    'left': filter_left,
    'right': filter_right,
    'both': filter_both,
    'teeth': filter_teeth,
    'eyebrows': filter_eyebrows,
}
############################################################

""" 1. CONNECT TO EEG STREAM """
# Search for active LSL streams
print('Looking for an EEG stream...')
streams = resolve_byprop('type', 'EEG', timeout=2)
if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')
print("Start acquiring data")
inlet = StreamInlet(streams[0], max_chunklen=128)
eeg_time_correction = inlet.time_correction()

n_timesteps = 128

while True:
    # inlet.pull_chunk(timeout=1.0, max_samples=128)
    eeg_data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=n_timesteps)
    eeg_data = np.array(eeg_data)[:, :-1]  # sample, channel

    ########################### Preprocessing ###################################
    x_eyebrows = pipeline(eeg_data, filters['eyebrows'], scalers['eyebrows'])
    x_left = pipeline(eeg_data, filters['left'], scalers['left'])
    x_right = pipeline(eeg_data, filters['right'], scalers['right'])
    x_both = pipeline(eeg_data, filters['both'], scalers['both'])
    x_teeth = pipeline(eeg_data, filters['teeth'], scalers['teeth'])
    x = np.concatenate(
        [
            x_eyebrows,
            x_left,
            x_right,
            x_both,
            x_teeth
        ],
        axis=1
    )
    
    input = np.concatenate([x], axis=1)
    input = np.expand_dims(input, 0)
    input = np.expand_dims(input, -1)
    input = input.transpose(0, 2, 1, 3)
    # print(input.shape)
    assert input.shape == (1, 20, 128, 1)
    #############################################################################


    ############################### Inference ###################################
    y_pred = model.predict([
        input[:, :4], 
        input[:, 4:8], 
        input[:, 8:12],
        input[:, 12:16],
        input[:, 16:20]
    ])
    y_pred = np.argmax(y_pred, 2)[0]
    #############################################################################
    print(y_pred)
    count = [0, 0, 0, 0, 0, 0]

    temp = np.unique(y_pred, return_counts=True)
    for i in range(temp[0].shape[0]):
        count[temp[0][i]] = temp[1][i]
    count[0] = 0
    pred = np.argmax(count)

    if pred == 1:
        print('Eyebrows - Go forward')
        ser.write(bytes('1', 'ascii'))
    elif pred == 2:
        print('Left - Turn left')
        ser.write(bytes('5', 'ascii'))
    elif pred == 3:
        print('Right - Turn right')
        ser.write(bytes('6', 'ascii'))
    elif pred == 4:
        print('Both - Stop')
        ser.write(bytes('0', 'ascii'))
    elif pred == 5:
        print('Teeth - Go backward')
        ser.write(bytes('2', 'ascii'))
