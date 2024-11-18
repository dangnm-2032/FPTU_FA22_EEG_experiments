from FPTU_FA24_EEG_Artifacts_Recognition.config import *
from FPTU_FA24_EEG_Artifacts_Recognition.constants import *
import joblib
import pandas as pd
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, config: ConfigurationManager) -> None:
        self.config = config

    def initialize_preprocess_module(self):
        self.filters = get_all_filters()

        self.scalers = {}
        config = self.config.get_dataset_config()
        for label in config.label:
            self.scalers[label] = joblib.load(Path(
                os.path.join(
                    config.scaler_path,
                    label + config.scaler_extension
                )
            ))


    def load_data(self):
        dataset_config = self.config.get_dataset_config()
        self.data = {}
        for label in dataset_config.label:
            files = []
            for idx, _, position, trial in itter_dataset_file_by_label(dataset_config, label):
                filepath = dataset_config.filepath_format.format_map(
                    {
                        'subject_id': idx,
                        'label': label,
                        'position': position,
                        'trial': trial
                    }
                )

                raw_data_filepath = Path(
                    os.path.join(dataset_config.raw_data_path, filepath)
                )
                roi_filepath = Path(
                    dataset_config.output_roi_path,
                    filepath
                )
                files.append([raw_data_filepath, roi_filepath])
            print(len(files))
            self.data[label] = deepcopy(files)

    def transform_data(self):
        self.input = {
            'eyebrows': [],
            'left': [],
            'right': [],
            'both': [],
            'teeth': []
        }
        self.output = []

        dataset_config = self.config.get_dataset_config()
        model_config = self.config.get_eeg_model_config()

        timestep = model_config.both.Samples

        for label in dataset_config.label:
            temp_input = []
            temp_output = []
            total_file = len(self.data[label])
            for i, (input_filepath, output_filepath) in enumerate(self.data[label]):
                print(label, i, total_file, end='\r')
                input_df = pd.read_csv(input_filepath).drop(columns=['timestamps', 'Right AUX'])
                output_df = pd.read_csv(output_filepath, index_col=0)

                input_data = input_df.to_numpy()
                output_data = output_df.to_numpy()

                assert input_data.shape[1] == 4
                assert output_data.shape[0] == input_data.shape[0]

                for i in range(0, input_data.shape[0] - timestep):
                    if 1 in output_data[i:i+timestep]:
                        window_input = input_data[i:i+timestep]
                        window_output = output_data[i:i+timestep]

                        for _label in dataset_config.label:
                            filter = self.filters[_label]
                            scaler = self.scalers[_label]
                            _input = window_input.copy()
                            for col in range(window_input.shape[1]):
                                _input[:, col] = filter(window_input[:, col])
                            _input = scaler.transform(_input) # sample, channel (64, 4)

                            self.input[_label].append(_input)
                        
                        self.output = window_output * LABEL2IDX[label]
            print()






    def train(self):
        pass