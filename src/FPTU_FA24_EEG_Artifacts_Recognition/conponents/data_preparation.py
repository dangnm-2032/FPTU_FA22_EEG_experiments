from FPTU_FA24_EEG_Artifacts_Recognition.config import ConfigurationManager
from FPTU_FA24_EEG_Artifacts_Recognition.logging import logger
from FPTU_FA24_EEG_Artifacts_Recognition.utils import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

class DataPreparer:
    def __init__(self, config: ConfigurationManager) -> None:
        self.config = config

    def raw_data_validation(self):
        # Get raw dataset config
        config = self.config.get_dataset_config()
        for idx, label, position, trial in itter_dataset_file(config):
            filepath = config.filepath_format.format_map(
                {
                    'subject_id': idx,
                    'label': label,
                    'position': position,
                    'trial': trial
                }
            )
            raw_data_filepath = Path(
                os.path.join(config.raw_data_path, filepath)
            )
            raw_roi_filepath = Path(
                os.path.join(config.raw_roi_path, filepath)
            )
            
            if not os.path.isfile(raw_data_filepath):
                logger.exception(f'File {raw_data_filepath} not found!')
                raise Exception(f'File {raw_data_filepath} not found!')

            if not os.path.isfile(raw_roi_filepath):
                logger.exception(f'File {raw_roi_filepath} not found!')
                raise Exception(f'File {raw_roi_filepath} not found!')
      

        logger.info('Raw data validation process finished!')
        del config, idx, label, position, trial, filepath, raw_data_filepath, raw_roi_filepath

    def transform_roi_files(self):
        config = self.config.get_dataset_config()
        os.makedirs(config.output_roi_path, exist_ok=True)

        config = self.config.get_dataset_config()

        for idx, label, position, trial in itter_dataset_file(config):
            filepath = config.filepath_format.format_map(
                {
                    'subject_id': idx,
                    'label': label,
                    'position': position,
                    'trial': trial
                }
            )
            raw_data_filepath = Path(
                os.path.join(config.raw_data_path, filepath)
            )

            raw_roi_filepath = Path(
                os.path.join(config.raw_roi_path, filepath)
            )

            output_roi_filepath = Path(
                config.output_roi_path,
                filepath
            )

            data_df = pd.read_csv(raw_data_filepath).drop(columns=['timestamps', 'Right AUX'])
            label_df = pd.read_csv(raw_roi_filepath).to_numpy()
            label_arr = np.zeros(len(data_df))
            
            for row in label_df:
                label_arr[int(row[0]):int(row[1])] = int(row[2])
            
            label_df = pd.DataFrame(label_arr)
            os.makedirs(os.path.dirname(output_roi_filepath), exist_ok=True)
            label_df.to_csv(output_roi_filepath)
        logger.info('Transforming ROI files successfully!')
        del config, idx, label, position, trial, filepath, raw_data_filepath, raw_roi_filepath, output_roi_filepath, data_df, label_df, label_arr, row

    def preparing_scaler(self, main_label):
        config = self.config.get_dataset_config()
        os.makedirs(config.scaler_path, exist_ok=True)
        big_data = []
        for idx, label, position, trial in itter_dataset_file(config):
            filepath = config.filepath_format.format_map(
                {
                    'subject_id': idx,
                    'label': label,
                    'position': position,
                    'trial': trial
                }
            )
            raw_data_filepath = Path(
                os.path.join(config.raw_data_path, filepath)
            )
            roi_filepath = Path(
                config.output_roi_path,
                filepath
            )

            data_df = pd.read_csv(raw_data_filepath).drop(columns=['timestamps', 'Right AUX'])
            label_df = pd.read_csv(roi_filepath, index_col=0)
            data_df['label'] = label_df['0']
            filtered_df = data_df[data_df['label'] == 1]
            filtered_df = filtered_df.drop(columns=['label'])
            data = filtered_df.to_numpy()
            filters = get_all_filters()
            for col in range(data.shape[1]):
                data[:, col] = filters[main_label](data[:, col])
            big_data.append(data.copy())

        big_data = np.concatenate(big_data)
        scaler = StandardScaler()
        scaler.fit(big_data)
        output_scaler_path = Path(
            os.path.join(config.scaler_path, main_label + config.scaler_extension)
        )
        joblib.dump(scaler, output_scaler_path)
        del config, big_data, idx, label, position, trial,filepath,raw_data_filepath,roi_filepath, output_scaler_path,data_df,label_df,filtered_df,data,filters,col,scaler

    def fitting_scaler_for_left(self):
        self.preparing_scaler('left')

    def fitting_scaler_for_right(self):
        self.preparing_scaler('right')

    def fitting_scaler_for_both(self):
        self.preparing_scaler('both')

    def fitting_scaler_for_teeth(self):
        self.preparing_scaler('teeth')

    def fitting_scaler_for_eyebrows(self):
        self.preparing_scaler('eyebrows')
