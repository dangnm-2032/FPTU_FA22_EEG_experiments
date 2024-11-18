from FPTU_FA24_EEG_Artifacts_Recognition.config import ConfigurationManager
from FPTU_FA24_EEG_Artifacts_Recognition.logging import logger
import os
from pathlib import Path

class DataPreparer:
    def __init__(self, config: ConfigurationManager) -> None:
        self.config = config

    def raw_data_validation(self):
        # Get raw dataset config
        config = self.config.get_dataset_config()
        num_subject = len(config.details)
        for idx in range(num_subject):
            subject = config.details[idx]
            for label in config.label:
                for position in range(subject.position):
                    for trial in range(subject.trial):
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
        logger.log('Raw data validatio process finished!')
        

    def transform_roi_files(self):
        pass

    def preprocess_for_left(self):
        pass

    def preprocess_for_right(self):
        pass

    def preprocess_for_both(self):
        pass

    def preprocess_for_teeth(self):
        pass

    def preprocess_for_eyebrows(self):
        pass