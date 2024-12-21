import flask
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from pipeline import CustomPipeline  # Ensure CustomPipeline is imported

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def make_predictions():
    if request.method == 'POST':
        # Collect input data
        input_data = {
            'ListingNumber': [int(request.form.get('ListingNumber'))],
            'BorrowerAPR': [float(request.form.get('BorrowerAPR'))],
            'BorrowerRate': [float(request.form.get('BorrowerRate'))],
            'LoanMonthsSinceOrigination': [int(request.form.get('LoanMonthsSinceOrigination'))],
            'LoanNumber': [int(request.form.get('LoanNumber'))],
            'LoanOriginationQuarter': [request.form.get('LoanOriginationQuarter')],
            'LP_CustomerPayments': [float(request.form.get('LP_CustomerPayments'))],
            'LP_CustomerPrincipalPayments': [float(request.form.get('LP_CustomerPrincipalPayments'))],
            'LoanCurrentDaysDelinquent': [int(request.form.get('LoanCurrentDaysDelinquent'))],
            'MonthlyLoanPayment': [float(request.form.get('MonthlyLoanPayment'))],
            'EmploymentStatus': [request.form.get('EmploymentStatus')]
        }

        # Convert input data to pandas DataFrame
        input_df = pd.DataFrame.from_dict(input_data)

        # Debug: print input_df to check if data is correctly formatted
        print("Input DataFrame:", input_df)

        # Make predictions
        y_class_pred, y_reg_pred = model.predict(input_df)

        # Debug: print predictions to check the output
        print("Classification Prediction:", y_class_pred)
        print("Regression Prediction:", y_reg_pred)

        # Handle prediction results
        class_response = 'have' if y_class_pred[0] == 1 else 'have not'

        # Set a threshold for the regression response
        threshold = 0  # You can adjust this threshold based on your domain knowledge and requirements
        reg_response = 'positive' if y_reg_pred > threshold else 'negative'

        return render_template('predictPage.html', class_response=class_response, reg_response=reg_response)

@app.route('/api')
def hello():
    response = {'MESSAGE': 'Welcome to the new API route'}
    return jsonify(response)

file_pkl = 'assets/combined_pipeline.pkl'

if __name__ == '__main__':
    model = joblib.load(file_pkl)  # Ensure the model file path is correct
    app.run(host='0.0.0.0', port=8000, debug=True)
