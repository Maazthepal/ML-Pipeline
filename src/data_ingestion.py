import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def Load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data from {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
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

def data_manipulation(df: pd.DataFrame) -> pd.DataFrame:
    """Perform data manipulation tasks."""
    try:
        df.drop('customerID', axis=1, inplace=True)
        logger.info("Dropped 'customerID' column")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        logger.info("Converted 'TotalCharges' to numeric")
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
        logger.info("Filled missing values in 'TotalCharges' with mean")
        return df
    
    except KeyError as e:
        logger.error(f"Key error during data manipulation: {e}")
        raise
    except Exception as e:  
        logger.error(f"Error during data manipulation: {e}")
        raise

def save_data(train_data: pd.DataFrame,test_data: pd.DataFrame, data_path: str,) -> None:
    """Save DataFrame tto a Train and Test datasets"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test_data.csv'), index=False)
        logger.info(f"Train and test data saved successfully in {raw_data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = "https://raw.githubusercontent.com/Maazthepal/ML-Pipeline/refs/heads/main/experiments/Churn.csv"
        df = Load_data(file_path= data_path)
        final_df = data_manipulation(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path="./data")
    except Exception as e:
        logger.error(f"Error in main data ingestion process: {e}")
        print(f"Error : {e}")

if __name__ == "__main__":
    main()