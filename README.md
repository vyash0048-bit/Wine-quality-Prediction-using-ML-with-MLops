# 🚀 ML Project with MLflow & DagsHub

An **end-to-end Machine Learning pipeline** built using industry best practices.  
This project covers the complete ML lifecycle — from data ingestion to model evaluation — with **experiment tracking using MLflow** and **remote logging via DagsHub**.

---

## 📌 Project Features

- Modular ML pipeline
- Configuration-driven architecture (YAML based)
- Data ingestion, validation, transformation
- Model training & evaluation
- Experiment tracking using **MLflow**
- Remote MLflow tracking via **DagsHub**
- Flask web application for UI
- Clean logging & exception handling

---


## 🗂️ Project Structure

```text
MLproject-with-MLflow/
├── config/
│   └── config.yaml
│
├── src/
│   └── mlProject/
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       │
│       ├── pipeline/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   └── stage_05_model_evaluation.py
│       │
│       ├── config/
│       │   └── configuration.py
│       │
│       ├── entity/
│       │   └── config_entity.py
│       │
│       ├── utils/
│       │   └── common.py
│       │
│       └── logger/
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── assets/
│       └── img/
│
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## 🛠️ How to Run the Project

### 🔹 STEP 1: Clone the Repository

git clone https://github.com/vyash0048-bit/MLproject-with-MLflow.git
cd MLproject-with-MLflow

### 🔹 STEP 2: Create Conda Environment
conda create -n mlproj python=3.8 -y
conda activate mlproj
### 🔹 STEP 3: Install Dependencies
pip install -r requirements.txt
### 🔹 STEP 4: Run ML Pipeline
python main.py
This will execute:

- Data Ingestion
- Data Validation
- Data Transformation
- Model Training
- Model Evaluation
- MLflow logging

### 🔹 STEP 5: Run Flask Web App
python app.py
Open in browser: http://127.0.0.1:8080

### 📊 MLflow Experiment Tracking
▶ Local MLflow UI (optional)
mlflow ui
Open: http://127.0.0.1:5000
Note: Local UI shows only local runs.

### 🌐 DagsHub Integration (Remote Tracking)
MLflow experiments are logged remotely to DagsHub.

### 🔑 Set Environment Variables (Windows)
setx MLFLOW_TRACKING_URI "https://dagshub.com/vyash0048/MLproject-with-MLflow.mlflow"
setx MLFLOW_TRACKING_USERNAME "vyash0048"
setx MLFLOW_TRACKING_PASSWORD "<YOUR_DAGSHUB_TOKEN>"
⚠️ Restart terminal after setting variables.

### 🔍 View Experiments on DagsHub
https://dagshub.com/vyash0048/MLproject-with-MLflow
→ Experiments → MLflow
🔐 Security Note
🚨 Never commit access tokens to GitHub.

Use environment variables

Add .env to .gitignore

Rotate tokens if exposed

## 🧠 Key Highlights
- Industry-grade ML project structure
- Configuration-driven pipelines
- MLflow + DagsHub integration
- Flask UI integration

## 🚀 Future Improvements
- Add DVC for data versioning
- Add FastAPI inference service
- Add CI/CD with GitHub Actions
- Cloud deployment (AWS / GCP)
