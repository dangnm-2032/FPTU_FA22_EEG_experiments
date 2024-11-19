from FPTU_FA24_EEG_Artifacts_Recognition.config import *
from FPTU_FA24_EEG_Artifacts_Recognition.pipeline import *

config_manager = ConfigurationManager()

pipeline = DataPreparationPipeline(config_manager)
pipeline.main()

pipeline = TrainingPipeline(config_manager)
pipeline.main()

pipeline = EvaluationPipeline(config_manager)
pipeline.main()