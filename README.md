Employee Attrition Prediction Project Documentation
1. Overview
The Employee Attrition Prediction project is designed to predict the likelihood of an employee leaving the company. This application utilizes various machine learning techniques to analyze employee data and make predictions. The project consists of several key components: data ingestion, data transformation, model training, exception handling, logging, utilities, and a web interface for user interaction.

2. Components
2.1 Data Ingestion
Purpose:
Handles reading the raw dataset, removing duplicates, renaming columns, and removing outliers.
Splits the data into training and testing sets and saves them in specified paths.
Key Methods:
remove_outliers_iqr:
Removes outliers from a specified column using the Interquartile Range (IQR) method.
initiate_data_ingestion:
Reads the data, removes duplicates and outliers, performs a train-test split, and saves the resulting datasets.
2.2 Data Transformation
Purpose:
Preprocesses the data by creating pipelines for numerical and categorical features and applying transformations such as imputation and scaling.
Ensures that both the training and testing datasets are transformed using the same preprocessing steps.
Key Methods:
get_data_transformer_object:
Creates and returns a ColumnTransformer object, which applies specific pipelines to numerical and categorical columns.
initiate_data_transformation:
Applies the preprocessing object to the training and testing data and returns the transformed datasets along with the preprocessor path and feature names.
2.3 Model Trainer
Purpose:
Trains multiple machine learning models using the transformed datasets and evaluates their performance.
Selects the best model based on the accuracy score and saves the trained model for future use.
Key Methods:
initiate_model_trainer:
Splits the data into features and target, evaluates different models with hyperparameter tuning, selects the best model, fits it to the training data, and returns the accuracy of the model on the test set.
2.4 Exception Handling
Purpose:
Handles exceptions that occur during the execution of the program and provides detailed error messages.
Key Classes/Methods:
error_message_detail(error, error_detail: sys):
Extracts details about the error, including the file name, line number, and error message.
CustomException:
A custom exception class that formats error messages using the error_message_detail function.
2.5 Logging
Purpose:
Captures logs during the execution of the program to track events, debug issues, and monitor performance.
Configuration:
Logs are stored in a logs directory, with filenames based on the current timestamp.
logging.basicConfig:
Configures the logging format, file location, and log level.
2.6 Utilities
Purpose:
Provides utility functions to save and load objects, and evaluate machine learning models.
Key Methods:
save_object(file_path, obj):
Saves a Python object to a specified file path using pickle.
load_object(file_path):
Loads a Python object from a specified file path using pickle.
evaluate_models(X_train, y_train, X_test, y_test, models, param):
Evaluates multiple models using GridSearchCV, fits the best model, and returns their performance scores.
2.7 Web Interface (app.py & HTML)
Purpose:

Provides a user-friendly interface to input employee data and predict attrition.
Files:

app.py:
A Flask application that serves the web interface, handles user input, and displays prediction results.
home.html:
The HTML file for the home page where users can input employee data and view prediction results.
Key Routes:

/ (Home Page):
Displays the main form for user input.
/predict_datapoint:
Handles form submission, processes input data, and displays the prediction result.
3. How to Run the Project
Install Dependencies:

Install the necessary Python packages using pip:
bash
Copy code
pip install -r requirements.txt
Run the Flask Application:

Start the Flask application:
bash
Copy code
python app.py
Access the Web Interface:

Open a web browser and navigate to http://localhost:5000/ to interact with the application.
4. Future Enhancements
Add more features and fine-tune the model for better accuracy.
Implement user authentication for secure access.
Add more visualization tools to better understand employee attrition patterns.
