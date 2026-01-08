import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

Logs_dir = "logs"
os.makedirs(Logs_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(Logs_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
               'PaperlessBilling', 'MultipleLines', 'OnlineSecurity', 
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies', 'Churn']

multi_cols = ['PaymentMethod', 'Contract', 'InternetService']

def preprocess_data(df: pd.DataFrame, ohe_list: list, le_list: list) -> pd.DataFrame:
    """"Preprocess the input DataFrame by handling values, encoding categorical variables"""
    logger.info("Starting data preprocessing")
    try:
        # Droping Duplicates
        df = df.drop_duplicates(keep='first')
        logger.info("Dropped duplicate rows from the DataFrame")
        
        df = df.replace({
    'No internet service': 'No',
    'No phone service': 'No'
    })
        logger.info("Replaced 'No internet service' and 'No phone service' with 'No'")

        # Label Encoding
        logger.info("Pre-Processing Started...")
        le = LabelEncoder()
        for col_nam in le_list:
            df[col_nam] = le.fit_transform(df[col_nam])
            logger.info("Label Encoding Implemented on %s", col_nam)
        logger.info("Label Encoding Successfull")

        # One-Hot Encoding
        pre_processed_df = pd.get_dummies(df, columns=ohe_list, drop_first=True, dtype=int)
        logger.info("One Hot Encoding Implemented...")
        return pre_processed_df
    
    except KeyError as e:
        logger.error(f"Key error during preprocessing: {e}")
        raise

    except Exception as e:
        logger.error(f"Error replacing service values: {e}")
        raise

def main():
    try:
        # Fetch the data
        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')

        # Transform the data
        train_processed_data = preprocess_data(train_data, ohe_list=multi_cols, le_list=binary_cols)
        test_processed_data = preprocess_data(test_data, ohe_list=multi_cols, le_list=binary_cols)

        # Store the Processed Data inside data/processed
        data_path = "./data/interim"
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug("Process data Saved to %s", data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 