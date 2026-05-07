import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.scaler = joblib.load(Path('artifacts/data_transformation/scaler.joblib'))

    
    def predict(self, data):
        # Convert incoming numpy array back to DataFrame for processing
        columns = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        df = pd.DataFrame(data, columns=columns)

        # 1. Feature engineering (Must match training)
        df['alcohol_to_acidity'] = df['alcohol'] / (df['volatile acidity'] + 1e-10)
        df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-10)

        # 2. Scale
        scaled_data = self.scaler.transform(df)

        # 3. Predict
        prediction = self.model.predict(scaled_data)

        return prediction