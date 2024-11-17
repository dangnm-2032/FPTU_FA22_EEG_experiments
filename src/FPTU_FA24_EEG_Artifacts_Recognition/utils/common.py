import yaml
from pathlib import Path
from box import ConfigBox 
from box.exceptions import BoxValueError
from FPTU_FA24_EEG_Artifacts_Recognition.logging import logger
from ensure import ensure_annotations
import os
from FPTU_FA24_EEG_Artifacts_Recognition.constants import *

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox: 
    """Read yaml file and returns ConfigBox instance

    Args:
        path_to_yaml: Path to yaml file
    
    Releases:
        ValueError: if yaml file is empty 
        e: empty file
    Returns:
        ConfigBox instance
    """
    try: 
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file '{path_to_yaml}' loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"yaml file: '{path_to_yaml}' is empty")
    except Exception as e: 
        raise e
    
def stage_name(
    text: str
) -> str:
    text = ">" * 10 + text + "<" * 10
    return text