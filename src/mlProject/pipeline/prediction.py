import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import os



class PredictionPipeline:
    def __init__(self):
        # Paths relative to this file's location to be robust to CWD changes
        # __file__ is src/mlProject/pipeline/prediction.py
        # root is 4 levels up
        self.root = Path(__file__).resolve().parent.parent.parent.parent
        
        model_path = self.root / 'artifacts/model_trainer/model.joblib'
        scaler_path = self.root / 'artifacts/data_transformation/scaler.joblib'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Current working directory: {os.getcwd()}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Current working directory: {os.getcwd()}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    
    def predict(self, data):
        # Convert incoming numpy array back to DataFrame for processing
        columns = [
            'fixed acidity', 'volatile acidity', 'citric acid',
            'chlorides', 'total sulfur dioxide', 'density',
            'sulphates', 'alcohol'
        ]
        df = pd.DataFrame(data, columns=columns)

        # 1. Feature engineering (Must match training)
        df['alcohol_to_acidity'] = df['alcohol'] / (df['volatile acidity'] + 1e-10)

        # 2. Scale
        scaled_data = self.scaler.transform(df)

        # 3. Predict
        prediction = self.model.predict(scaled_data)

        return prediction