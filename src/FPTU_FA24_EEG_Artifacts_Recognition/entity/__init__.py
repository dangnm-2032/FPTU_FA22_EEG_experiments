from dataclasses import dataclass
from pathlib import Path

@dataclass
class Dataset:
    raw_data_path: Path
    raw_roi_path: Path
