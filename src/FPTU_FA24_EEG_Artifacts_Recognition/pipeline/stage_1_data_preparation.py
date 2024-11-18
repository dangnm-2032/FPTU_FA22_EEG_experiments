from FPTU_FA24_EEG_Artifacts_Recognition.conponents import *
from FPTU_FA24_EEG_Artifacts_Recognition.config import ConfigurationManager

class DataPreparationPipeline:
    def __init__(self, configuration_manager: ConfigurationManager) -> None:
        self.data_preparer = DataPreparer(configuration_manager)

    def main(self):
        # Check if raw data are enough as config, as well as theirs ROI
        self.data_preparer.raw_data_validation()

        # Transform raw ROI file
        self.data_preparer.transform_roi_files()

        # Initialize preprocess pipeline for each label
        self.data_preparer.preprocess_for_left()
        self.data_preparer.preprocess_for_right()
        self.data_preparer.preprocess_for_both()
        self.data_preparer.preprocess_for_teeth()
        self.data_preparer.preprocess_for_eyebrows()

if __name__ == '__main__':
    config_manager = ConfigurationManager()
    pipeline = DataPreparationPipeline(config_manager)
    pipeline.main()