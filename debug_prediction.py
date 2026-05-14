import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys

# Force local src import
sys.path.insert(0, os.path.abspath('src'))

from mlProject.pipeline.prediction import PredictionPipeline

# Load some test data
raw_data = pd.read_csv('artifacts/data_ingestion/winequality-red.csv')

# Use only the features expected by the updated pipeline
expected_cols = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'chlorides', 'total sulfur dioxide', 'density',
    'sulphates', 'alcohol'
]

pipeline = PredictionPipeline()

print("Testing Prediction Pipeline with Reduced Features:")

def test_samples(samples, label):
    print(f"\nTesting {label} samples:")
    for i, row in samples.iterrows():
        data = row[expected_cols].values.reshape(1, 8)
        pred = pipeline.predict(data)
        print(f"Sample {i} -> Predicted: {pred}")

# Get samples from different classes
low_samples = raw_data[raw_data['quality'] <= 4].head(3)
med_samples = raw_data[(raw_data['quality'] >= 5) & (raw_data['quality'] <= 6)].head(3)
high_samples = raw_data[raw_data['quality'] >= 7].head(3)

test_samples(low_samples, "LOW")
test_samples(med_samples, "MEDIUM")
test_samples(high_samples, "HIGH")
