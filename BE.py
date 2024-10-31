from muselsl.stream import stream
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import serial
import time
import multiprocessing as mp
import numpy as np
import subprocess
import sys

from tensorflow.keras.models import load_model
from preprocessing import *
from utils import *





import warnings 
warnings.filterwarnings('ignore') 

# stream(
#     address=None,
#     retries=5,
# )

# Status
# 1. Muse connection
# 2. Check if Muse signal is stable
# Menu
# 1. List COM ports
# 2. Test connection to Controller
# 3. Start software by doing a set of movements

class ControlBackend:
    def __init__(self) -> None:
        self.status_muse_connect = False
        self.status_signal_stable = False
        self.status_controller_connect = False
        self.status_muse_stream = False

        self.list_ports = {}
        self.controller_port = None
        self.ser = None
        self.muse_stream = None
        self.inlet = None

        self.n_timesteps = 64
        

    def list_all_ports(self):
        self.list_ports = []
        list_ports = serial.tools.list_ports.comports()
        for i, (port, desc, _) in enumerate(sorted(list_ports)):
            print("{}. {}: {}".format(i+1, port, desc))
            self.list_ports.append(port)
        print('0. Back')
        return self.list_ports

    def choose_port(self):
        while True:
            choice = input('Choose port: ')
            try:
                choice = int(choice)
            except:
                print('Please input a number')
                continue

            if choice > len(self.list_ports):
                print('Your selection is not exists!')
                continue
            
            if choice != 0:
                self.controller_port = self.list_ports[choice-1]
            break

    def connect_controller(self):
        if self.controller_port is None:
            print('Please choose COM port!')
            return 
        
        if self.ser is None:
            try:
                self.ser = serial.Serial(
                    port=self.controller_port,
                    baudrate=9600,
                    timeout=1,
                    write_timeout=1
                )
                print('Connected!')
            except:
                print('Wrong port')
                self.ser = None
                return
            time.sleep(1)
        
        time.sleep(1)
        try:
            self.ser.write(bytes('#', 'ascii'))
        except:
            print('Wrong port')
            self.ser.close()
            self.ser = None
            return
        ret = self.ser.read().decode('ascii')
        if ret != '@':
            print('Wrong port')
            self.ser.close()
            self.ser = None
            return

        self.status_controller_connect = True
        print("Connection successfully!")

    def check_muse_stream(self):
        if self.muse_stream is None:
            self.muse_stream = subprocess.Popen([sys.executable, '-m', 'muselsl', 'stream'])
            print(self.muse_stream.pid)

        if self.muse_stream is not None:
            # self.status_muse_stream = self.muse_stream.is_alive()
            self.status_muse_stream = True if self.muse_stream.poll() is None else False
        
        if self.status_muse_stream is False:
            self.muse_stream = None
            if self.inlet:
                self.status_muse_connect = False
                self.inlet.close_stream()
                self.inlet = None

    def connect_muse_stream(self):
        if not self.inlet:
            # Search for active LSL streams
            print('Looking for an EEG stream...')
            streams = resolve_byprop('type', 'EEG', timeout=2)
            if len(streams) == 0:
                print('Can\'t find EEG stream.')
                return
        
        
            print("Start acquiring data")
            self.inlet = StreamInlet(streams[0], max_chunklen=self.n_timesteps)
            self.status_muse_connect = True

    def exit(self):
        if self.inlet:
            self.inlet.close_stream()
        
        if self.ser:
            self.ser.close()

        if self.status_muse_stream:
            self.muse_stream.kill()

        exit()

    def init_module(self):
        ##################### SCALER #########################################
        import joblib
        label_name = ['eyebrows', 'left', 'right', 'both', 'teeth']
        self.scalers = {}
        for label in label_name:
            self.scalers[label] = joblib.load(rf'.\pipeline_{label}\checkpoints\scaler_standard.save')
        ######################################################################

        ####################### MODEL ############################################
        self.model = load_model(r'.\checkpoints\orthogonal_standard_64_timesteps_trainable_True.keras')
        ####################### END MODEL ########################################

        ####### FILTER #############################################
        self.filters = {
            'left': filter_left,
            'right': filter_right,
            'both': filter_both,
            'teeth': filter_teeth,
            'eyebrows': filter_eyebrows,
        }
        ############################################################

    def inference(self):
        eeg_data, _ = self.inlet.pull_chunk(
            timeout=1.0, 
            max_samples=self.n_timesteps
        )
        eeg_data = np.array(eeg_data)[:, :-1]  # sample, channel

        ########################### Preprocessing ###################################
        x_eyebrows = pipeline(eeg_data, self.filters['eyebrows'], self.scalers['eyebrows'])
        x_left = pipeline(eeg_data, self.filters['left'], self.scalers['left'])
        x_right = pipeline(eeg_data, self.filters['right'], self.scalers['right'])
        x_both = pipeline(eeg_data, self.filters['both'], self.scalers['both'])
        x_teeth = pipeline(eeg_data, self.filters['teeth'], self.scalers['teeth'])
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
        assert input.shape == (1, 20, self.n_timesteps, 1)
        #############################################################################


        ############################### Inference ###################################
        y_pred = self.model.predict([
            input[:, :4], 
            input[:, 4:8], 
            input[:, 8:12],
            input[:, 12:16],
            input[:, 16:20]
        ])
        y_pred = np.argmax(y_pred, 2)[0]
        #############################################################################
        # print(y_pred)
        count = [0, 0, 0, 0, 0, 0]

        temp = np.unique(y_pred, return_counts=True)
        for i in range(temp[0].shape[0]):
            count[temp[0][i]] = temp[1][i]
        count = np.array(count)
        # count[0] = 0

        # print(count)
        pred = np.argmax(count)

        if pred == 1:
            print('Eyebrows - Go backward')
            self.ser.write(bytes('2', 'ascii'))
        elif pred == 2:
            print('Left - Turn left')
            self.ser.write(bytes('5', 'ascii'))
        elif pred == 3:
            print('Right - Turn right')
            self.ser.write(bytes('6', 'ascii'))
        elif pred == 4:
            print('Both - Stop')
            self.ser.write(bytes('0', 'ascii'))
        elif pred == 5:
            print('Teeth - Go forward')
            self.ser.write(bytes('1', 'ascii'))

        return pred

    def main(self):
        while True:
            self.check_muse_stream()
            MENU = f"""
Status:
    - Muse stream: {self.status_muse_stream}
    - Muse connection: {self.status_muse_connect}
    - Check if Muse signal is stable: {self.status_signal_stable}
    - Controller connection: {self.status_controller_connect}
    - Current COM port: {self.controller_port}
Menu:
    1. Choose COM port
    2. Connection to Controller
    3. Muse connection
    4. Start 
    0. Exit
"""
            print(MENU)
            choice = input("Choose: ")
            try:
                choice = int(choice)
            except:
                print('Please input a number')
                continue

            if choice == 1:
                self.list_all_ports()
                self.choose_port()
            elif choice == 2:
                self.connect_controller()
            elif choice == 3:
                self.connect_muse_stream()
            elif choice == 4:
                try:
                    self.init_module()
                    while True:
                        pred = self.inference()
                        print(pred)
                except:
                    print("Lost connection")
            elif choice == 0:
                self.exit()
            else:
                print("Not implemented")
                continue

if __name__ == '__main__':
    app = ControlBackend()
    app.main()