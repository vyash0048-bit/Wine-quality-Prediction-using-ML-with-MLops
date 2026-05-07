import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, f1
    


    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

     
        mlflow.set_tracking_uri(self.config.mlflow_uri)

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            accuracy, f1 = self.eval_metrics(test_y, predicted_qualities)

            scores = {"accuracy": accuracy, "f1_score": f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="XGBoostModel"
            )
