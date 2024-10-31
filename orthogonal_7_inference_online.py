import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

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

import warnings
warnings.filterwarnings("ignore")

############# PREDICTION RESULT PLOT #############################
plt.ion()

plot_duration = 2 # seconds
SR = 256
x = np.linspace(0, plot_duration, SR*plot_duration)

fig = plt.figure(figsize=(10, 8))

temp = int(SR * plot_duration / 2)
max = 1
min = 0

line_TP9_raw, = plt.plot(x, [0, 1,]  * temp)
line_AF7_raw, = plt.plot(x, [0, 1,]  * temp)
line_AF8_raw, = plt.plot(x, [0, 1,]  * temp)
line_TP10_raw, = plt.plot(x, [0, 1,]  * temp)

line_TP9_filter, = plt.plot(x, [0, 1,]  * temp)
line_AF7_filter, = plt.plot(x, [0, 1,]  * temp)
line_AF8_filter, = plt.plot(x, [0, 1,]  * temp)
line_TP10_filter, = plt.plot(x, [0, 1,]  * temp)


line_eyebrows, = plt.plot(x, [0, 1,] * temp, label='eyebrows')
line_left, = plt.plot(x, [0, 1,] * temp, label='left')
line_right, = plt.plot(x, [0, 1,] * temp, label='right')
line_both, = plt.plot(x, [0, 1,] * temp, label='both')
line_teeth, = plt.plot(x, [0, 1,] * temp, label='teeth')
plt.ylim(-2, 12.5)
plt.legend(loc=1)
plt.axis('off')
plt.text(-0.2, -0.1, 'PREDICT')
plt.text(-0.2, 10.8, 'TP9')
plt.text(-0.2, 8.8, 'AF7')
plt.text(-0.2, 6.8, 'AF8')
plt.text(-0.2, 4.8, 'TP10')
buffer = np.zeros((SR*plot_duration, 13))
##################################################################


def pipeline(x, filter, scaler):
    x_new = x.copy()
    for col in range(x_new.shape[1]):
        x_new[:, col] = filter(x_new[:, col])
    x_new = scaler.transform(x_new)

    return x_new


##################### MODEL ###############################################
n_timesteps = 64
model = load_model(r'.\checkpoints\orthogonal_64_timesteps_trainable_True.keras')
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
inlet = StreamInlet(streams[0], max_chunklen=n_timesteps)
eeg_time_correction = inlet.time_correction()
###########################################################################




record_data = []

try:
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
        assert input.shape == (1, 20, n_timesteps, 1)
        #############################################################################


        ############################### Inference ###################################
        y_pred = model.predict([
            input[:, :4], 
            input[:, 4:8], 
            input[:, 8:12],
            input[:, 12:16],
            input[:, 16:20]
        ])

        #############################################################################
        print(y_pred.shape)

        buffer[:-n_timesteps] = buffer[n_timesteps:]

        buffer[-n_timesteps:, 0] = eeg_data[:, 0] /200 + 11 # TP9
        buffer[-n_timesteps:, 1] = input[0, 4, :, 0] * 10 + 5 # TP9 filter

        buffer[-n_timesteps:, 2] = eeg_data[:, 1] /200 + 9 # AF7
        buffer[-n_timesteps:, 3] = input[0, 5, :, 0] * 10 + 4.5 # AF7 filter

        buffer[-n_timesteps:, 4] = eeg_data[:, 2] /200 + 7 # AF8
        buffer[-n_timesteps:, 5] = input[0, 6, :, 0] * 10 + 2.8 # AF8 filter

        buffer[-n_timesteps:, 6] = eeg_data[:, 3] /200 + 5 # TP10
        buffer[-n_timesteps:, 7] = input[0, 7, :, 0] * 10 - 3 # TP10 filter

        buffer[-n_timesteps:, 8] = y_pred[0, :, 1] + 0
        buffer[-n_timesteps:, 9] = y_pred[0, :, 2] + 0
        buffer[-n_timesteps:, 10] = y_pred[0, :, 3] + 0
        buffer[-n_timesteps:, 11] = y_pred[0, :, 4] + 0
        buffer[-n_timesteps:, 12] = y_pred[0, :, 5] + 0
        
        line_TP9_raw.set_ydata(buffer[:, 0])
        line_TP9_filter.set_ydata(buffer[:, 1])

        line_AF7_raw.set_ydata(buffer[:, 2])
        line_AF7_filter.set_ydata(buffer[:, 3])

        line_AF8_raw.set_ydata(buffer[:, 4])
        line_AF8_filter.set_ydata(buffer[:, 5])

        line_TP10_raw.set_ydata(buffer[:, 6])
        line_TP10_filter.set_ydata(buffer[:, 7])

        line_eyebrows.set_ydata(buffer[:, 8])
        line_left.set_ydata(buffer[:, 9])
        line_right.set_ydata(buffer[:, 10])
        line_both.set_ydata(buffer[:, 11])
        line_teeth.set_ydata(buffer[:, 12])

        # record_data.append(
        #     np.concatenate(
        #         [
        #             eeg_data,
        #             x,
        #             y_pred[0, :, 1][:, np.newaxis],
        #             y_pred[0, :, 2][:, np.newaxis],
        #             y_pred[0, :, 3][:, np.newaxis],
        #             y_pred[0, :, 4][:, np.newaxis],
        #             y_pred[0, :, 5][:, np.newaxis]
        #         ],
        #         axis=1
        #     )
        # )

        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    df = pd.DataFrame(
        np.concatenate(record_data, axis=0), 
        columns=[
            'Raw TP9',
            'Raw AF7',
            'Raw AF8',
            'Raw TP10',
            'Eyebrows TP9',
            'Eyebrows AF7',
            'Eyebrows AF8',
            'Eyebrows TP10',
            'Left TP9',
            'Left AF7',
            'Left AF8',
            'Left TP10',
            'Right TP9',
            'Right AF7',
            'Right AF8',
            'Right TP10',
            'Both TP9',
            'Both AF7',
            'Both AF8',
            'Both TP10',
            'Teeth TP9',
            'Teeth AF7',
            'Teeth AF8',
            'Teeth TP10',
            'Predict Eyebrows',
            'Predict Left',
            'Predict Right',
            'Predict Both',
            'Predict Teeth',
        ])
    df.to_csv(f'Inference_Record_{int(time.time())}.csv')