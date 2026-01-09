import os
import pickle
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score
import numpy as np
import pandas as pd
import logging
import json
import yaml

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(filepath: str):
    """Load Model From the file path"""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully from %s", filepath)
        return model
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error occurred while loading the model: %s", e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise
        
def load_data(filepath: str):
    """Load Data From the data file path"""
    try:
        df = pd.read_csv(filepath)
        logger.debug("Data successfully loaded from %s (Shape: %s)", filepath, df.shape)
        return df
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error occurred while loading data: %s", e)
        raise

def evaluate_model(model, X_test, y_test, threshold:int):
    """Evaluate the Model and return the Evaluation Metrics"""
    try:
        logger.info("Starting model evaluation...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': float(accuracy),
            'recall': float(recall),
            'precision': float(precision),
            'auc': float(roc)
        }

        logger.info("Model Evaluation Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  AUC:       {roc:.4f}")
        
        return metrics_dict
    except Exception as e:
        logger.error("Unexpected error occurred while evaluating the model: %s", e)
        raise

def save_metrics(file_path: str, metrics: dict):
    """Save the Model Evaluation Metrics to a JSON File"""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics successfully saved to {file_path}")
        
    except Exception as e:
        logger.error("Unexpected error occurred while saving the metrics: %s", e)
        raise

def main():
    try:
        params = load_params('params.yaml')['model_evaluation']

        logger.info("=" * 60)
        logger.info("Starting Model Evaluation Pipeline")
        logger.info("=" * 60)
        
        logger.info("Loading model...")
        model = load_model("./models/xgb_model.pkl")

        logger.info("Loading test data...")
        X_test = load_data("./data/processed/X_test.csv")
        y_test = load_data("./data/processed/y_test.csv")
        
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values.ravel()
        
        logger.info(f"Test data loaded: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

        metrics = evaluate_model(model, X_test, y_test, threshold=params['threshold'])

        save_metrics("reports/metrics.json", metrics)
        
        logger.info("=" * 60)
        logger.info("Model Evaluation Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("Unexpected error occurred during model evaluation process: %s", e)
        raise

if __name__ == "__main__":
    main()