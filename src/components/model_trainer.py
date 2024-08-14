import os
import sys
import warnings
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning, NotFittedError

import xgboost as xgb
import lightgbm as lgb

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Suppress ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Define models to be evaluated
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(solver='liblinear'),  # 'liblinear' for 'l1' penalty
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                "LightGBM": lgb.LGBMClassifier()
            }
            
            # Define hyperparameters for each model
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 10, 20, 30]
                },
                "Logistic Regression": {
                    'penalty': ['l2'],  # 'l1' is removed for 'liblinear' solver
                    'C': [0.1, 1, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "LightGBM": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }

            # Add a Voting Classifier with all the models
            voting_clf = VotingClassifier(estimators=[(name, model) for name, model in models.items()], voting='soft')
            models["Voting Classifier"] = voting_clf

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))
            # Get the best model name from the report
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Log the best model and its score
            logging.info(f"Best model: {best_model_name} with a score of {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with an acceptable score", sys)

            logging.info("Best model found on both training and testing dataset")

            # Fit the best model before saving
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            try:
                predicted = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, predicted)
                return accuracy
            except NotFittedError:
                raise CustomException("The model is not fitted. Ensure model fitting was successful.", sys)

        except Exception as e:
            raise CustomException(str(e), sys)
