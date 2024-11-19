from FPTU_FA24_EEG_Artifacts_Recognition.config import *
from FPTU_FA24_EEG_Artifacts_Recognition.pipeline import *

def main():
    config_manager = ConfigurationManager()

    try:
        STAGE_NAME = stage_name("STAGE 1: DATA PREPARATION")
        logger.info(STAGE_NAME)
        pipeline = DataPreparationPipeline(config_manager)
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        STAGE_NAME = stage_name("STAGE 2: TRAINING")
        logger.info(STAGE_NAME)
        pipeline = TrainingPipeline(config_manager)
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        STAGE_NAME = stage_name("STAGE 3: EVALUATION")
        logger.info(STAGE_NAME)
        pipeline = EvaluationPipeline(config_manager)
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
    main()