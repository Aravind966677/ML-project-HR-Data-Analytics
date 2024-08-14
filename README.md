# Employee Attrition Prediction Project

## Overview
The Employee Attrition Prediction project is designed to predict the likelihood of an employee leaving the company. It utilizes various machine learning techniques to analyze employee data and provide predictions. This project consists of components for data ingestion, data transformation, model training, exception handling, logging, and a web interface.

## Project Components

### 1. Data Ingestion
- **Purpose**: Handles reading the raw dataset, removing duplicates, renaming columns, and removing outliers. It splits the data into training and testing sets and saves them in specified paths.
- **Key Methods**:
  - `remove_outliers_iqr`: Removes outliers using the Interquartile Range (IQR) method.
  - `initiate_data_ingestion`: Reads, cleans, splits, and saves the dataset.

### 2. Data Transformation
- **Purpose**: Responsible for preprocessing the data. It creates pipelines for numerical and categorical features and applies transformations like imputation and scaling. It ensures consistent preprocessing of both training and testing datasets.
- **Key Methods**:
  - `get_data_transformer_object`: Creates a ColumnTransformer for numerical and categorical columns.
  - `initiate_data_transformation`: Applies preprocessing and returns the transformed datasets, preprocessor path, and feature names.

### 3. Model Trainer
- **Purpose**: Trains multiple machine learning models, evaluates their performance, and selects the best model based on accuracy. Saves the trained model for future use.
- **Key Methods**:
  - `initiate_model_trainer`: Splits data into features and target, evaluates models with hyperparameter tuning, selects the best model, and fits it to the training data.

### 4. Exception Handling
- **Purpose**: Handles exceptions during execution and provides detailed error messages.
- **Key Classes/Methods**:
  - `error_message_detail(error, error_detail: sys)`: Extracts error details including file name, line number, and error message.
  - `CustomException`: Custom exception class for formatted error messages.

### 5. Logging
- **Purpose**: Captures logs for tracking events, debugging issues, and monitoring performance.
- **Configuration**: Logs are stored in a `logs` directory with timestamps. Configured using `logging.basicConfig`.

### 6. Utilities
- **Purpose**: Provides functions to save and load objects and evaluate machine learning models.
- **Key Methods**:
  - `save_object(file_path, obj)`: Saves an object using `pickle`.
  - `load_object(file_path)`: Loads an object using `pickle`.
  - `evaluate_models(X_train, y_train, X_test, y_test, models, param)`: Evaluates models and returns performance scores.

### 7. Web Interface
- **Files**:
  - `app.py`: Flask application that serves the web interface, handles user input, and displays predictions.
  - `home.html`: HTML file for user input and displaying prediction results.
- **Key Routes**:
  - `/` (Home Page): Displays the main form.
  - `/predict_datapoint`: Handles form submission and displays prediction results.


