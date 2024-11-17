from pathlib import Path
from FPTU_FA24_EEG_Artifacts_Recognition.constants import *
from FPTU_FA24_EEG_Artifacts_Recognition.utils import *
from FPTU_FA24_EEG_Artifacts_Recognition.entity import *

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_abc_config(self):
        pass