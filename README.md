---
title: Wine Quality Prediction AI
emoji: 🍷
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# 🍷 Wine Quality Prediction AI

An **end-to-end Machine Learning pipeline** for predicting wine quality, built with a modular architecture and deployed to **Hugging Face Spaces**.

This project implements a streamlined 8-feature prediction model using **XGBoost**, featuring automated data transformation, SMOTE for class balancing, and a polished Flask web interface.

---

## 📌 Project Features

- **Streamlined Feature Set**: Optimized using correlation analysis (8 key chemical features).
- **Advanced Preprocessing**: Automated handling of outliers, class imbalance (SMOTE), and scaling.
- **XGBoost Classifier**: High-performance gradient boosting model.
- **Modular Pipeline**: Clean separation of ingestion, validation, transformation, training, and evaluation.
- **Flask Web Interface**: Modern, responsive UI for easy quality prediction.
- **Hugging Face Deployment**: Automated CI/CD deployment via GitHub Actions.
- **Robust Error Handling**: Real-time validation for non-numeric user inputs.

---

## 🧪 Chemical Features Used (Reduced Set)

We optimized the model by dropping low-correlation features (*residual sugar, free sulfur dioxide, pH*) to focus on:
1. **Fixed Acidity**
2. **Volatile Acidity**
3. **Citric Acid**
4. **Chlorides**
5. **Total Sulfur Dioxide**
6. **Density**
7. **Sulphates**
8. **Alcohol**

---

## 🗂️ Project Structure

```text
Wine-quality-Prediction/
├── src/mlProject/
│   ├── components/      # Modular logic for each ML stage
│   ├── pipeline/        # Stage execution scripts
│   ├── config/          # Configuration management
│   └── entity/          # Data structure definitions
├── templates/           # Flask HTML templates
├── static/              # CSS, JS, and Assets
├── artifacts/           # Generated data, models, and scalers
├── app.py               # Flask application
├── main.py              # Pipeline execution script
├── Dockerfile           # Hugging Face deployment config
└── requirements.txt     # Python dependencies
```

---

## 🛠️ How to Run Locally

### 🔹 1. Clone the Repository
```bash
git clone https://github.com/vyash0048-bit/Wine-quality-Prediction-using-ML-with-MLops.git
cd Wine-quality-Prediction-using-ML-with-MLops
```

### 🔹 2. Create Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 🔹 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 4. Execute ML Pipeline (Retrain)
```bash
python main.py
```
This runs the full lifecycle: Ingestion → Validation → Transformation → Training → Evaluation.

### 🔹 5. Start Web App
```bash
python app.py
```
Open: `http://127.0.0.1:8080`

---

## 🚀 Deployment

The project is deployed on **Hugging Face Spaces** using Docker.

- **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/YashAI07/Wine-quality-Prediction)
- **CI/CD**: Any push to the `main` branch automatically triggers a GitHub Action to sync the repository with the Hugging Face Hub.

---

## 🧠 Technical Highlights
- **Architecture**: Modular and configuration-driven (YAML).
- **Class Balancing**: SMOTE used during transformation to handle minority quality classes.
- **Persistence**: Models and scalers saved as `.joblib` for efficient inference.
- **Scalability**: Designed to easily add more features or swap models in `params.yaml`.
