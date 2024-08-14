import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the data transformation pipeline.
        '''
        try:
            self.numerical_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 
                                  'average_monthly_hours', 'tenure', 'work_accident', 
                                  'promotion_last_5years']
            self.categorical_columns = ['department', 'salary']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first'))  # Use drop='first' to avoid multicollinearity
                ]
            )

            logging.info(f"Categorical columns: {self.categorical_columns}")
            logging.info(f"Numerical columns: {self.numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, self.numerical_columns),
                    ("cat_pipeline", cat_pipeline, self.categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function is responsible for transforming the training and testing data.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "left"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Transform input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Extract and print column names from one-hot encoding
            cat_pipeline = preprocessing_obj.named_transformers_['cat_pipeline']
            one_hot_encoder = cat_pipeline.named_steps['one_hot_encoder']
            cat_feature_names = one_hot_encoder.get_feature_names_out(self.categorical_columns)

            all_feature_names = self.numerical_columns + list(cat_feature_names) + [target_column_name]
            logging.info(f"Feature names after encoding: {all_feature_names}")

            # Optionally, print or save feature names to a file for inspection
            feature_names_df = pd.DataFrame(columns=all_feature_names)
            logging.info(f"Feature names DataFrame:\n{feature_names_df.head()}")

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                all_feature_names
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path, feature_names = data_transformation.initiate_data_transformation(
        "path_to_train.csv", 
        "path_to_test.csv"
    )
    print(f"Feature names after encoding: {feature_names}")
