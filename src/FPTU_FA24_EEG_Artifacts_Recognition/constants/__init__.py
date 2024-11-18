from pathlib import Path
import os

CONFIG_FILE_PATH = Path('config/config.yaml')
PARAMS_FILE_PATH = Path('config/params.yaml')
DATASET_DETAIL_FILE_PATH = Path('config/dataset.yaml')

CURRENT_WORKING_DIRECTORY = os.getcwd()

LABEL2IDX = {
    'eyebrows': 1,
    'left': 2,
    'right': 3,
    'both': 4,
    'teeth': 5,
}