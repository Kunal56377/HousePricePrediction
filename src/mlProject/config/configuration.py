from mlProject.constants import *
from mlProject.utils.common import *
from mlProject.entity.config_entity import *
from sklearn.pipeline import make_pipeline
from mlProject.preprocesser.preprocesser import Preprocessor
from dataclasses import dataclass
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor




class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH,
        pipelines_filepath = PIPELINES_FILE_PATH
        ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        self.pipelines = read_yaml(pipelines_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    
    def get_model_Pipeline(self) -> ModelPipelineConfig:
        pipelines = self.pipelines.pipelines

        print(list(self.pipelines.keys())[1:])
        pipelines_pre = {
                    'ridge': make_pipeline(Preprocessor(),eval(pipelines.ridge)), 
                    'rf': make_pipeline(Preprocessor(),eval(pipelines.rf)), 
                    'gb': make_pipeline(Preprocessor(),eval(pipelines.gb)), 
                    'xg': make_pipeline(Preprocessor(), eval(pipelines.xg))
                }
        
        grid_pre = {}
        for algo in list(self.pipelines.keys())[1:]:
            params = {} 

            for parm in self.pipelines[algo].keys():
                params[parm] = list(self.pipelines[algo][parm])
        
            grid_pre[algo] = params
            
        # print(grid_pre)
        model_Pipe_line_Config = ModelPipelineConfig(
            pipelines= pipelines_pre,
            grid= grid_pre
        )
        return  model_Pipe_line_Config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ElasticNet
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        pipelines_obj = self.get_model_Pipeline()
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            pipelines = pipelines_obj.pipelines,
            grid =pipelines_obj.grid ,
            target_column = schema.name
        )

        return model_trainer_config
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage  started <<<<<<")
        obj = ConfigurationManager()
        pipe = obj.get_model_Pipeline()
        print(pipe.grid)
        #logger.info(f">>>>>> stage {pipe} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
