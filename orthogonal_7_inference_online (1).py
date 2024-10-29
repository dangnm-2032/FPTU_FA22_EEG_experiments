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
from matplotlib.animation import FuncAnimation
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

fig, axs = plt.subplots(5, 1, figsize=(12, 8))  

temp = int(SR * plot_duration / 2)


line_TP9_raw, = axs[0].plot(x, [0, 1] * temp, label='TP9 raw')
line_TP9_filter, = axs[0].plot(x, [0, 1] * temp, label='TP9 filtered')
axs[0].set_ylim(-200, 200)
axs[0].legend(loc='upper right')


line_AF7_raw, = axs[1].plot(x, [0, 1] * temp, label='AF7 raw')
line_AF7_filter, = axs[1].plot(x, [0, 1] * temp, label='AF7 filtered')
axs[1].set_ylim(-200, 200)
axs[1].legend(loc='upper right')


line_AF8_raw, = axs[2].plot(x, [0, 1] * temp, label='AF8 raw')
line_AF8_filter, = axs[2].plot(x, [0, 1] * temp, label='AF8 filtered')
axs[2].set_ylim(-200, 200)
axs[2].legend(loc='upper right')


line_TP10_raw, = axs[3].plot(x, [0, 1] * temp, label='TP10 raw')
line_TP10_filter, = axs[3].plot(x, [0, 1] * temp, label='TP10 filtered')
axs[3].set_ylim(-200, 200)
axs[3].legend(loc='upper right')



line_eyebrows, = axs[4].plot(x, [0, 1,] * temp, label='eyebrows',alpha = 0.7)
line_left, = axs[4].plot(x, [0, 1,] * temp, label='left',alpha = 0.7)
line_right, = axs[4].plot(x, [0, 1,] * temp, label='right',alpha = 0.7)
line_both, = axs[4].plot(x, [0, 1,] * temp, label='both',alpha = 0.7)
line_teeth, = axs[4].plot(x, [0, 1,] * temp, label='teeth',alpha = 0.7)
axs[4].legend(loc='upper right')
axs[4].set_ylim(0, 12)


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
model = load_model(r'.\checkpoints\orthogonal_standard_64_timesteps_trainable_True.keras')
###########################################################################


##################### SCALER ##############################################
import joblib
label_name = ['eyebrows', 'left', 'right', 'both', 'teeth']
scalers = {}
for label in label_name:
    scalers[label] = joblib.load(rf'.\pipeline_{label}\checkpoints\scaler_standard.save')
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




# record_data = []

# try:
#     while True:
#         # inlet.pull_chunk(timeout=1.0, max_samples=128)
#         eeg_data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=n_timesteps)
#         eeg_data = np.array(eeg_data)[:, :-1]  # sample, channel

#         ########################### Preprocessing ###################################
#         x_eyebrows = pipeline(eeg_data, filters['eyebrows'], scalers['eyebrows'])
#         x_left = pipeline(eeg_data, filters['left'], scalers['left'])
#         x_right = pipeline(eeg_data, filters['right'], scalers['right'])
#         x_both = pipeline(eeg_data, filters['both'], scalers['both'])
#         x_teeth = pipeline(eeg_data, filters['teeth'], scalers['teeth'])
#         x = np.concatenate(
#             [
#                 x_eyebrows,
#                 x_left,
#                 x_right,
#                 x_both,
#                 x_teeth
#             ],
#             axis=1
#         )
        
#         input = np.concatenate([x], axis=1)
#         input = np.expand_dims(input, 0)
#         input = np.expand_dims(input, -1)
#         input = input.transpose(0, 2, 1, 3)
#         # print(input.shape)
#         assert input.shape == (1, 20, n_timesteps, 1)
#         #############################################################################


#         ############################### Inference ###################################
#         y_pred = model.predict([
#             input[:, :4], 
#             input[:, 4:8], 
#             input[:, 8:12],
#             input[:, 12:16],
#             input[:, 16:20]
#         ])

#         #############################################################################
#         print(y_pred.shape)

#         buffer[:-n_timesteps] = buffer[n_timesteps:]

#         buffer[-n_timesteps:, 0] = eeg_data[:, 0] # TP9
#         buffer[-n_timesteps:, 1] = input[0, 0, :, 0]*10 + 17 # TP9 filter

#         buffer[-n_timesteps:, 2] = eeg_data[:, 1] # AF7
#         buffer[-n_timesteps:, 3] = input[0, 1, :, 0]*10   + 15 # AF7 filter

#         buffer[-n_timesteps:, 4] = eeg_data[:, 2] # AF8
#         buffer[-n_timesteps:, 5] = input[0, 2, :, 0]*10 + 13 # AF8 filter

#         buffer[-n_timesteps:, 6] = eeg_data[:, 3] # TP10
#         buffer[-n_timesteps:, 7] = input[0, 3, :, 0]*10  + 11# TP10 filter

#         buffer[-n_timesteps:, 8] = y_pred[0, :, 1]*5 +5
#         buffer[-n_timesteps:, 9] = y_pred[0, :, 2]*5 +10
#         buffer[-n_timesteps:, 10] = y_pred[0, :, 3]*5 +15
#         buffer[-n_timesteps:, 11] = y_pred[0, :, 4]*5 +20
#         buffer[-n_timesteps:, 12] = y_pred[0, :, 5]*5 +25
        
#         line_TP9_raw.set_ydata(buffer[:, 0])
#         line_TP9_filter.set_ydata(buffer[:, 1])

#         line_AF7_raw.set_ydata(buffer[:, 2])
#         line_AF7_filter.set_ydata(buffer[:, 3])

#         line_AF8_raw.set_ydata(buffer[:, 4])
#         line_AF8_filter.set_ydata(buffer[:, 5])

#         line_TP10_raw.set_ydata(buffer[:, 6])
#         line_TP10_filter.set_ydata(buffer[:, 7])

#         line_eyebrows.set_ydata(buffer[:, 8])
#         line_left.set_ydata(buffer[:, 9])
#         line_right.set_ydata(buffer[:, 10])
#         line_both.set_ydata(buffer[:, 11])
#         line_teeth.set_ydata(buffer[:, 12])

#         # record_data.append(
#         #     np.concatenate(
#         #         [
#         #             eeg_data,
#         #             x,
#         #             y_pred[0, :, 1][:, np.newaxis],
#         #             y_pred[0, :, 2][:, np.newaxis],
#         #             y_pred[0, :, 3][:, np.newaxis],
#         #             y_pred[0, :, 4][:, np.newaxis],
#         #             y_pred[0, :, 5][:, np.newaxis]
#         #         ],
#         #         axis=1
#         #     )
#         # )


#         fig.canvas.draw()
#         fig.canvas.flush_events()
def update_plot(frame):

    eeg_data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=n_timesteps)
    eeg_data = np.array(eeg_data)[:, :-1] 

    # Preprocess data
    x_eyebrows = pipeline(eeg_data, filters['eyebrows'], scalers['eyebrows'])
    x_left = pipeline(eeg_data, filters['left'], scalers['left'])
    x_right = pipeline(eeg_data, filters['right'], scalers['right'])
    x_both = pipeline(eeg_data, filters['both'], scalers['both'])
    x_teeth = pipeline(eeg_data, filters['teeth'], scalers['teeth'])
    
  
    x = np.concatenate([x_eyebrows, x_left, x_right, x_both, x_teeth], axis=1)
    input = np.expand_dims(x, axis=(0, -1)).transpose(0, 2, 1, 3)


    y_pred = model.predict([
        input[:, :4],
        input[:, 4:8],
        input[:, 8:12],
        input[:, 12:16],
        input[:, 16:20]
    ])

 
    buffer[:-n_timesteps] = buffer[n_timesteps:]
    buffer[-n_timesteps:, 0] = eeg_data[:, 0]  # TP9
    buffer[-n_timesteps:, 1] = input[0, 0, :, 0] * 10 + 17  # TP9 filter
    buffer[-n_timesteps:, 2] = eeg_data[:, 1]  # AF7
    buffer[-n_timesteps:, 3] = input[0, 1, :, 0] * 10 + 15  # AF7 filter
    buffer[-n_timesteps:, 4] = eeg_data[:, 2]  # AF8
    buffer[-n_timesteps:, 5] = input[0, 2, :, 0] * 10 + 13  # AF8 filter
    buffer[-n_timesteps:, 6] = eeg_data[:, 3]  # TP10
    buffer[-n_timesteps:, 7] = input[0, 3, :, 0] * 10 + 11  # TP10 filter
    buffer[-n_timesteps:, 8] = y_pred[0, :, 1]  + 2
    buffer[-n_timesteps:, 9] = y_pred[0, :, 2]  + 4
    buffer[-n_timesteps:, 10] = y_pred[0, :, 3]  + 6
    buffer[-n_timesteps:, 11] = y_pred[0, :, 4]  + 8
    buffer[-n_timesteps:, 12] = y_pred[0, :, 5]  + 10

    # Update the plot
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

    return line_TP9_raw, line_TP9_filter, line_AF7_raw, line_AF7_filter, line_AF8_raw, line_AF8_filter, line_TP10_raw, line_TP10_filter, line_eyebrows, line_left, line_right, line_both, line_teeth

ani = FuncAnimation(fig, update_plot, interval=1000 / SR)  

plt.show()
try: 
    plt.show(block=True)
except KeyboardInterrupt:
    pass
    # df = pd.DataFrame(
    #     np.concatenate(record_data, axis=0), 
    #     columns=[
    #         'Raw TP9',
    #         'Raw AF7',
    #         'Raw AF8',
    #         'Raw TP10',
    #         'Eyebrows TP9',
    #         'Eyebrows AF7',
    #         'Eyebrows AF8',
    #         'Eyebrows TP10',
    #         'Left TP9',
    #         'Left AF7',
    #         'Left AF8',
    #         'Left TP10',
    #         'Right TP9',
    #         'Right AF7',
    #         'Right AF8',
    #         'Right TP10',
    #         'Both TP9',
    #         'Both AF7',
    #         'Both AF8',
    #         'Both TP10',
    #         'Teeth TP9',
    #         'Teeth AF7',
    #         'Teeth AF8',
    #         'Teeth TP10',
    #         'Predict Eyebrows',
    #         'Predict Left',
    #         'Predict Right',
    #         'Predict Both',
    #         'Predict Teeth',
    #     ])
    # df.to_csv(f'Inference_Record_{int(time.time())}.csv')


