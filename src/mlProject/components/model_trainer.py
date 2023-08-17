import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        
        train_x = train_data.drop([self.config.target_column], axis=1)
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]
        
        print(test_x.info())
        fit_models = {}
        for algo, pipeline in self.config.pipelines.items(): 
            try: 
                print(algo)
                model = GridSearchCV(pipeline, self.config.grid[algo], n_jobs=-1, cv=10, scoring='r2',error_score='raise')
                model.fit(train_x,train_y)
                fit_models[algo] = model 
            except Exception as e: 
                print(f'Model {algo} had an error {e}')

        for algo,model in fit_models.items():
            model_name = f"model_{algo}.joblib"
            joblib.dump(model, os.path.join(self.config.root_dir,model_name))