import os
from mlProject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
from mlProject.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    def train_test_spliting(self):
        df = pd.read_csv(self.config.data_path)

        # 1. Load & clean
        df = df.drop_duplicates()

        # 2. Remove outliers (Disabled - often removes the very samples needed to identify High/Low quality)
        # for col in ['total sulfur dioxide', 'chlorides', 'residual sugar']:
        #     Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        #     IQR = Q3 - Q1
        #     df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

        # 3. Feature engineering
        df['alcohol_to_acidity'] = df['alcohol'] / df['volatile acidity']
        df['sulfur_ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']

        # 4. Bin quality (0: <=4, 1: 5-6, 2: >=7)
        df['quality'] = df['quality'].apply(lambda q: 0 if q <= 4 else (2 if q >= 7 else 1))

        # 5. Split FIRST
        X = df.drop('quality', axis=1)
        y = df['quality']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 6. Scale (fit only on train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler for prediction
        joblib.dump(scaler, os.path.join(self.config.root_dir, self.config.scaler_name))

        # 7. SMOTE only on training data
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        # Reconstruct DataFrames to save as CSV
        train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
        train_resampled['quality'] = y_train_resampled

        test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        test_scaled['quality'] = y_test.values

        train_resampled.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_scaled.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Advanced data transformation (User Recipe) completed.")