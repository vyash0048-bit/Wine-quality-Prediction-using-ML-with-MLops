import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys

# Add src to path
sys.path.append(os.path.abspath('src'))

from mlProject.pipeline.prediction import PredictionPipeline

# Load some test data
raw_data = pd.read_csv('artifacts/data_ingestion/winequality-red.csv')

# Get samples from different classes
low_samples = raw_data[raw_data['quality'] <= 4].head(3).drop('quality', axis=1)
med_samples = raw_data[(raw_data['quality'] >= 5) & (raw_data['quality'] <= 6)].head(3).drop('quality', axis=1)
high_samples = raw_data[raw_data['quality'] >= 7].head(3).drop('quality', axis=1)

pipeline = PredictionPipeline()

print("Testing Prediction Pipeline:")

def test_samples(samples, label):
    print(f"\nTesting {label} samples:")
    for i, row in samples.iterrows():
        data = row.values.reshape(1, 11)
        pred = pipeline.predict(data)
        print(f"Sample {i} -> Predicted: {pred}")

test_samples(low_samples, "LOW")
test_samples(med_samples, "MEDIUM")
test_samples(high_samples, "HIGH")

print("\nTesting EXTREME HIGH quality:")
extreme_high = np.array([[7.0, 0.1, 0.5, 2.0, 0.05, 15.0, 30.0, 0.99, 3.3, 0.9, 15.0]])
print(f"Extreme High -> Predicted: {pipeline.predict(extreme_high)}")

print("\nTesting EXTREME LOW quality:")
extreme_low = np.array([[7.0, 1.5, 0.0, 10.0, 0.2, 5.0, 100.0, 1.0, 3.8, 0.2, 8.0]])
print(f"Extreme Low -> Predicted: {pipeline.predict(extreme_low)}")
