from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Load the model
model = joblib.load('loan_prediction_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Home route to serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from the request
    features = np.array([data['ApplicantIncome'], data['CoapplicantIncome'], 
                         data['LoanAmount'], data['Loan_Amount_Term'], 
                         data['Credit_History'], data['Gender'], data['Married'], 
                         data['Dependents'], data['Education'], data['Self_Employed'], 
                         data['Property_Area']]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return the prediction result as a JSON response
    return jsonify({'Loan_Status': 'Approved' if prediction[0] == 1 else 'Rejected'})

if __name__ == '__main__':
    app.run(debug=True)
