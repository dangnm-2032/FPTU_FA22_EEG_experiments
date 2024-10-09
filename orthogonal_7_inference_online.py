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

import warnings
warnings.filterwarnings("ignore")

############# PREDICTION RESULT PLOT #############################
plt.ion()
fig = plt.figure(figsize=(10, 8))

plot_eyebrows = fig.add_subplot(511)
plot_left = fig.add_subplot(512)
plot_right = fig.add_subplot(513)
plot_both = fig.add_subplot(514)
plot_teeth = fig.add_subplot(515)

plot_eyebrows.set_title("Eyebrows")
plot_left.set_title("Left")
plot_right.set_title("Right")
plot_both.set_title("Both")
plot_teeth.set_title("Teeth")

line_eyebrows, = plot_eyebrows.plot(list(range(128)), [0, 1,] * 64)
line_left, = plot_left.plot(list(range(128)), [0, 1,] * 64)
line_right, = plot_right.plot(list(range(128)), [0, 1,] * 64)
line_both, = plot_both.plot(list(range(128)), [0, 1,] * 64)
line_teeth, = plot_teeth.plot(list(range(128)), [0, 1,] * 64)
##################################################################


def pipeline(x, filter, scaler):
    x_new = x.copy()
    for col in range(x_new.shape[1]):
        x_new[:, col] = filter(x_new[:, col])
    x_new = scaler.transform(x_new)

    return x_new


##################### MODEL ###############################################
model = load_model(r'.\checkpoints\orthogonal_2.keras')
###########################################################################


##################### SCALER ##############################################
import joblib
label_name = ['eyebrows', 'left', 'right', 'both', 'teeth']
scalers = {}
for label in label_name:
    scalers[label] = joblib.load(rf'.\pipeline_{label}\checkpoints\scaler.save')
###########################################################################

filters = {
    'left': filter_left,
    'right': filter_right,
    'both': filter_both,
    'teeth': filter_teeth,
    'eyebrows': filter_eyebrows,
}

##################### CONNECT EEG STREAM ##################################
print('Looking for an EEG stream...')
streams = resolve_byprop('type', 'EEG', timeout=2)
if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')
print("Start acquiring data")
inlet = StreamInlet(streams[0], max_chunklen=128)
eeg_time_correction = inlet.time_correction()
###########################################################################


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
    x_teeth = pipeline(eeg_data[:, [0, 3]], filters['teeth'], scalers['teeth'])
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
    assert input.shape == (1, 18, 128, 1)
    #############################################################################


    ############################### Inference ###################################
    y_pred = model.predict([
        input[:, :4], 
        input[:, 4:8], 
        input[:, 8:12],
        input[:, 12:16],
        input[:, 16:20]
    ])
    # y_pred = np.argmax(y_pred, 2)[0]
    #############################################################################
    print(y_pred.shape)
    
    line_eyebrows.set_ydata(y_pred[0, :, 1])
    line_left.set_ydata(y_pred[0, :, 2])
    line_right.set_ydata(y_pred[0, :, 3])
    line_both.set_ydata(y_pred[0, :, 4])
    line_teeth.set_ydata(y_pred[0, :, 5])

    fig.canvas.draw()
    fig.canvas.flush_events()