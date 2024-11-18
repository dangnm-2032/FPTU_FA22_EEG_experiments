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

def itter_dataset_file(config):
    num_subject = len(config.details)
    for idx in range(num_subject):
        subject = config.details[idx]
        for label in config.label:
            for position in range(subject.position):
                for trial in range(subject.trial):
                    yield idx, label, position, trial

def itter_dataset_file_by_label(config, label):
    num_subject = len(config.details)
    for idx in range(num_subject):
        subject = config.details[idx]
        for position in range(subject.position):
            for trial in range(subject.trial):
                yield idx, label, position, trial

def pipeline(x, filter, scaler, i=None, time_step=None):
    if i is None:
        i = 0
    if time_step is None:
        time_step = x.shape[0]
    x_new = x[i:i+time_step].copy()
    for col in range(x_new.shape[1]):
        x_new[:, col] = filter(x_new[:, col])
    x_new = scaler.transform(x_new)

    return x_new