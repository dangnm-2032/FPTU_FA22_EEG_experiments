from FPTU_FA24_EEG_Artifacts_Recognition.config import ConfigurationManager

class DataPreparer:
    def __init__(self, config: ConfigurationManager) -> None:
        self.config = config

    def raw_data_validation(self):
        pass

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