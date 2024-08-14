import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            # Load model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Preprocess features
            data_scaled = preprocessor.transform(features)
            print("Transformed Data:", data_scaled)
            
            # Make prediction
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 satisfaction_level: float,
                 last_evaluation: float,
                 number_project: int,
                 average_monthly_hours: float,
                 tenure: float,
                 work_accident: int,
                 promotion_last_5years: int,
                 department: str,
                 salary: str):
        
        self.satisfaction_level = satisfaction_level
        self.last_evaluation = last_evaluation
        self.number_project = number_project
        self.average_monthly_hours = average_monthly_hours
        self.tenure = tenure
        self.work_accident = work_accident
        self.promotion_last_5years = promotion_last_5years
        self.department = department
        self.salary = salary

    def get_data_as_data_frame(self):
        try:
            # Convert categorical features to appropriate format
            custom_data_input_dict = {
                "satisfaction_level": [self.satisfaction_level],
                "last_evaluation": [self.last_evaluation],
                "number_project": [self.number_project],
                "average_monthly_hours": [self.average_monthly_hours],
                "tenure": [self.tenure],
                "work_accident": [self.work_accident],
                "promotion_last_5years": [self.promotion_last_5years],
                "department": [self.department],
                "salary": [self.salary],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
