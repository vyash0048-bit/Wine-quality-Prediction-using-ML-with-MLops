import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import os



class PredictionPipeline:
    def __init__(self):
        # 1. Try path relative to this file (robust for local and container)
        self.root = Path(__file__).resolve().parent.parent.parent.parent
        model_path = self.root / 'artifacts/model_trainer/model.joblib'
        scaler_path = self.root / 'artifacts/data_transformation/scaler.joblib'
        
        # 2. Fallback to CWD-based path
        if not model_path.exists():
            model_path = Path('artifacts/model_trainer/model.joblib')
        if not scaler_path.exists():
            scaler_path = Path('artifacts/data_transformation/scaler.joblib')

        # Diagnostic info if still missing
        if not model_path.exists() or not scaler_path.exists():
            error_msg = f"Artifacts missing!\nLooking for model at: {model_path.absolute()}\nLooking for scaler at: {scaler_path.absolute()}\nCWD: {os.getcwd()}\n"
            # List current directory contents for debugging
            try:
                error_msg += f"Files in CWD: {os.listdir('.')}\n"
                if os.path.exists('artifacts'):
                    error_msg += f"Files in artifacts/: {os.listdir('artifacts')}\n"
            except:
                pass
            raise FileNotFoundError(error_msg)

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