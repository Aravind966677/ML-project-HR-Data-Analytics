import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def remove_outliers_iqr(self, df, column):
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the range for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('data/data.csv')
            logging.info('Read the dataset as dataframe')

            # Print total duplicates before removing
            total_duplicates_before = df.duplicated().sum()
            logging.info(f"Total duplicates before removal: {total_duplicates_before}")

            # Rename columns as needed
            df = df.rename(columns={
                'Work_accident': 'work_accident',
                'average_montly_hours': 'average_monthly_hours',
                'time_spend_company': 'tenure',
                'Department': 'department'
            })

            # Drop duplicate rows
            df = df.drop_duplicates()

            # Print total duplicates after removing
            total_duplicates_after = df.duplicated().sum()
            logging.info(f"Total duplicates after removal: {total_duplicates_after}")

            # Remove outliers in 'tenure' column
            df = self.remove_outliers_iqr(df, 'tenure')

            # Save the raw DataFrame with updated columns and duplicates removed
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path, feature_names = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    accuracy = modeltrainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Model Accuracy: {accuracy}")
