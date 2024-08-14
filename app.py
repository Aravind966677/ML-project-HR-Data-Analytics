from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = CustomData(
            satisfaction_level=float(request.form.get('satisfaction_level')),
            last_evaluation=float(request.form.get('last_evaluation')),
            number_project=int(request.form.get('number_project')),
            average_monthly_hours=float(request.form.get('average_monthly_hours')),
            tenure=float(request.form.get('tenure')),
            work_accident=int(request.form.get('work_accident')),
            promotion_last_5years=int(request.form.get('promotion_last_5years')),
            department=request.form.get('department'),
            salary=request.form.get('salary')
        )
        
        pred_df = data.get_data_as_data_frame()
        print("Data for Prediction:\n", pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction Results:", results)
        
        return render_template('home.html', results=results[0])
    
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
