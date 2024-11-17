from FPTU_FA24_EEG_Artifacts_Recognition.conponents import *

class DataPreparationPipeline:
    def __init__(self) -> None:
        self.data_preparer = DataPreparer()

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