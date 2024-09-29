# vital_analysis_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_ml_model(dataset_path):
    # Load dataset using the provided path
    df = pd.read_csv(dataset_path)

    # Define your required columns
    required_columns = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 
                        'Oxygen Saturation', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Risk Category']
    
    # Check if required columns exist
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataset.")

    # Prepare features and target variable
    X = df[['Heart Rate', 'Respiratory Rate', 'Body Temperature', 
            'Oxygen Saturation', 'Systolic Blood Pressure', 'Diastolic Blood Pressure']]
    y = df['Risk Category']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')

    return model

def predict_risk_with_ml(model, heart_rate, respiratory_rate, body_temperature, 
                          oxygen_saturation, systolic_blood_pressure, diastolic_blood_pressure):
    input_data = [[heart_rate, respiratory_rate, body_temperature, 
                   oxygen_saturation, systolic_blood_pressure, diastolic_blood_pressure]]

    # Make the prediction
    prediction = model.predict(input_data)

    return prediction[0]  # Return the numerical prediction (0 or 1)

def generate_issues_report(risk_category):
    issues = {
        "Low Risk": "No immediate concerns detected. Maintain a healthy lifestyle.",
        "High Risk": "Possible health issues detected. Consider consulting a healthcare provider. Conditions may include hypertension, respiratory distress, or other cardiovascular issues."
    }

    with open('patient_issues_report.txt', 'w') as file:
        file.write(f"Predicted Risk Category: {risk_category}\n")
        file.write("Health Issues:\n")
        file.write(issues[risk_category] + "\n")
    
    return f"Issues report generated as 'patient_issues_report.txt'."
