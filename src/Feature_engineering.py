import pandas as pd
import os
import logging

# Logger Setup
Logs_dir = "logs"
os.makedirs(Logs_dir, exist_ok=True)

logger = logging.getLogger("Feature_engineering")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(Logs_dir, 'Feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data: {file_path} is empty")
        raise
    except pd.errors.ParserError:
        logger.error(f"Parsing error: Could not parse data from {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def feature_engineering(train_data, test_data):
    """Perform feature engineering on train and test data"""
    try:
        logger.info("Starting feature engineering process")
        X_train = train_data.drop('Churn', axis=1)
        y_train = train_data['Churn']
        X_test = test_data.drop('Churn', axis=1)
        y_test = test_data['Churn']
        logger.info("Feature engineering process completed successfully")
        return X_train, y_train, X_test, y_test
    except KeyError as e:
        logger.error(f"Key error during feature engineering: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str):
    """Save DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    try:
        test_data=load_data('./data/interim/test_processed.csv')
        train_data=load_data('./data/interim/train_processed.csv')
        logger.info("Loaded train and test data for feature engineering")

        logger.info("Performing feature engineering...")
        X_train, y_train, X_test, y_test = feature_engineering(train_data, test_data)
        save_data(X_train,os.path.join('./data', 'processed', 'X_train.csv'))
        save_data(y_train,os.path.join('./data', 'processed', 'y_train.csv'))
        save_data(X_test,os.path.join('./data', 'processed', 'X_test.csv'))
        save_data(y_test,os.path.join('./data', 'processed', 'y_test.csv'))
        logger.info("Feature engineering completed and data saved.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print("An error occurred during feature engineering: %s" % e)

if __name__ == "__main__":
    main()