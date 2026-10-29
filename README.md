# Predictive Maintenance Using Machine Learning

## Project Overview
An end-to-end machine learning project using engine data to develop and evaluate predictive maintenance models.

## Objectives
- Validate and explore the dataset.
- Prepare data for machine learning.
- Compare models and tune hyperparameters.
- Evaluate model performance.
- Provide predictions through a Streamlit application.
- Automate the workflow using GitHub Actions.

## Project Structure
- `src/` — data validation, preparation, training, and evaluation scripts.
- `models/` — trained model artifacts.
- `artifacts/` — model comparison and evaluation metrics.
- `reports/` — evaluation figures and reports.
- `app.py` — Streamlit application.
- `requirements.txt` — dependencies.
- `.github/workflows/` — CI workflow.

## Model Evaluation
The evaluation metrics and model comparison results are available in the `artifacts/` folder.

## Running the Application
Install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Note
This project is for educational and demonstration purposes. Predictions should not be used as the sole basis for real-world maintenance decisions.
