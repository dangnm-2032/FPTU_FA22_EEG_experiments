from FPTU_FA24_EEG_Artifacts_Recognition.config import *
from FPTU_FA24_EEG_Artifacts_Recognition.conponents import *

class TrainingPipeline:
    def __init__(self, config: ConfigurationManager) -> None:
        self.trainer = Trainer(config)
    
    def main(self):
        self.trainer.initialize_preprocess_module()
        self.trainer.load_data()
        self.trainer.transform_data()
        self.trainer.train()

if __name__ == '__main__':
    config_manager = ConfigurationManager()
    pipeline = TrainingPipeline(config_manager)
    pipeline.main()