from FPTU_FA24_EEG_Artifacts_Recognition.config import *
from FPTU_FA24_EEG_Artifacts_Recognition.conponents import *

class EvaluationPipeline:
    def __init__(self, config: ConfigurationManager) -> None:
        self.validator = Validator(config)
    
    def main(self):
        self.validator.load_test_data()
        self.validator.load_model()
        self.validator.evaluate()

if __name__ == '__main__':
    config_manager = ConfigurationManager()
    pipeline = EvaluationPipeline(config_manager)
    pipeline.main()