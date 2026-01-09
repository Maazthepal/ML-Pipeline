import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
import logging
import yaml

# Set up logging
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("model_building")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(logs_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_file(file_path: str):
    """Load a pickled file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data: {file_path} is empty")
        raise e
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise e
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise e
    
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
    
def train_model(X_train, y_train, params:dict) -> XGBClassifier:
    """Train an XGBoost Classifier model."""
    try:
        if X_train.shape[0] != y_train.shape[0]:
            logger.error("Mismatched shapes between X_train and y_train")
            raise ValueError("Mismatched shapes between X_train and y_train")
        logger.info("Starting model training...")
        model = XGBClassifier(n_estimators=params['n_estimators'], 
                              learning_rate=params['learning_rate'],
                              max_depth=params['max_depth'], 
                              scale_pos_weight=params['scale_pos_weight'], 
                              random_state=params['random_state'],
                              subsample=params['subsample'],
                              colsample_bytree=params['colsample_bytree'],
                              )
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully !!!")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model, file_path: str):
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {file_path}")
    except FileNotFoundError as e:
        logger.error(f"Directory not found for saving model: {file_path}")
        raise e
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise e
    
def main():
    """Main function to build and save the model."""
    try:
            params = load_params('params.yaml')['model_building']
            
            X_train = load_file('data/processed/X_train.csv')
            y_train = load_file('data/processed/y_train.csv')
            model = train_model(X_train, y_train, params)

            model_path = 'models/xgb_model.pkl'
            save_model(model, model_path)

    except Exception as e:
            logger.error('Error Occured in the model building function: %s', e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()