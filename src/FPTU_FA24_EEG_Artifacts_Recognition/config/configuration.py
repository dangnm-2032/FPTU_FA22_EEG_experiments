from pathlib import Path
from vinewschatbot.constants import *
from vinewschatbot.utils import *
from vinewschatbot.entity import *

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)