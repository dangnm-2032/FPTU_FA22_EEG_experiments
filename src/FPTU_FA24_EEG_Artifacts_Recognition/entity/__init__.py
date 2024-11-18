from dataclasses import dataclass
from pathlib import Path

@dataclass
class Dataset:
    raw_data_path: Path
    raw_roi_path: Path

@dataclass
class RightModule:
    nb_classes: int
    Chans: int
    Samples: int
    dropoutRate: float
    kernLength: int
    F1: int
    D: int
    F2: int
    dropoutType: str

@dataclass
class RightModule:
    nb_classes: int
    Chans: int
    Samples: int
    dropoutRate: float
    kernLength: int
    F1: int
    D: int
    F2: int
    dropoutType: str

@dataclass
class EEGModule:
    nb_classes: int
    Chans: int
    Samples: int
    dropoutRate: float
    kernLength: int
    F1: int
    D: int
    F2: int
    dropoutType: str

@dataclass
class EEGModel:
    right: EEGModule
    left: EEGModule
    teeth: EEGModule
    both: EEGModule
    eyebrows: EEGModule

@dataclass
class RecordSubject:
    id: int
    name: str
    position: int
    trial: int

@dataclass
class Dataset:
    raw_data_path: Path
    raw_roi_path: Path
    output_data_path: Path
    output_roi_path: Path
    filepath_format: str
    label: list
    details: dict