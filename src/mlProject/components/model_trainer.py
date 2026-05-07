import pandas as pd
import os
from mlProject import logger
from xgboost import XGBClassifier
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.utils.class_weight import compute_sample_weight



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

        # Compute sample weights to handle imbalance directly in XGBoost
        sample_weights = compute_sample_weight(class_weight='balanced', y=train_y)


        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
        xgb.fit(train_x, train_y, sample_weight=sample_weights)

        joblib.dump(xgb, os.path.join(self.config.root_dir, self.config.model_name))

