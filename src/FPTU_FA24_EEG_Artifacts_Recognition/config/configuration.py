from pathlib import Path
from FPTU_FA24_EEG_Artifacts_Recognition.constants import *
from FPTU_FA24_EEG_Artifacts_Recognition.utils import *
from FPTU_FA24_EEG_Artifacts_Recognition.entity import *

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        dataset_details_filepath: Path = DATASET_DETAIL_FILE_PATH
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.dataset_details = read_yaml(dataset_details_filepath)

    def get_dataset_config(self):
        config = self.config.dataset
        details = {}
        dataset_detail = self.dataset_details.subject
        for idx in dataset_detail:
            temp = RecordSubject(
                id=idx,
                name=dataset_detail[idx].name,
                position=dataset_detail[idx].position,
                trial=dataset_detail[idx].trial
            )
            details[idx] = temp

        return Dataset(
            raw_data_path=config.raw_data_path, 
            raw_roi_path=config.raw_roi_path,
            output_data_path=config.output_data_path,
            output_roi_path=config.output_roi_path,
            filepath_format=config.filepath_format,
            label=config.label,
            details=details
        )

    def get_eeg_model_config(self):
        config = self.params.eeg_model
        left_module = EEGModule(
            nb_classes=config.left.nb_classes,
            Chans=config.left.Chans,
            Samples=config.left.Samples,
            dropoutRate=config.left.dropoutRate,
            kernLength=config.left.kernLength,
            F1=config.left.F1,
            D=config.left.D,
            F2=config.left.F2,
            dropoutType=config.left.dropoutType,
        )
        right_module = EEGModule(
            nb_classes=config.right.nb_classes,
            Chans=config.right.Chans,
            Samples=config.right.Samples,
            dropoutRate=config.right.dropoutRate,
            kernLength=config.right.kernLength,
            F1=config.right.F1,
            D=config.right.D,
            F2=config.right.F2,
            dropoutType=config.right.dropoutType,
        )
        both_module = EEGModule(
            nb_classes=config.both.nb_classes,
            Chans=config.both.Chans,
            Samples=config.both.Samples,
            dropoutRate=config.both.dropoutRate,
            kernLength=config.both.kernLength,
            F1=config.both.F1,
            D=config.both.D,
            F2=config.both.F2,
            dropoutType=config.both.dropoutType,
        )
        eyebrows_module = EEGModule(
            nb_classes=config.eyebrows.nb_classes,
            Chans=config.eyebrows.Chans,
            Samples=config.eyebrows.Samples,
            dropoutRate=config.eyebrows.dropoutRate,
            kernLength=config.eyebrows.kernLength,
            F1=config.eyebrows.F1,
            D=config.eyebrows.D,
            F2=config.eyebrows.F2,
            dropoutType=config.eyebrows.dropoutType,
        )
        teeth_module = EEGModule(
            nb_classes=config.teeth.nb_classes,
            Chans=config.teeth.Chans,
            Samples=config.teeth.Samples,
            dropoutRate=config.teeth.dropoutRate,
            kernLength=config.teeth.kernLength,
            F1=config.teeth.F1,
            D=config.teeth.D,
            F2=config.teeth.F2,
            dropoutType=config.teeth.dropoutType,
        )
        return EEGModel(
            left=left_module,
            right=right_module,
            both=both_module,
            teeth=teeth_module,
            eyebrows=eyebrows_module
        )