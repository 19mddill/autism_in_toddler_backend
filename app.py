
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and encoders
with open('model.pkl', 'rb') as model_file:
    model, label_encoders = pickle.load(model_file)

# Preprocess input JSON
def preprocess_input(data):
    # List all expected features in the correct order
    expected_features = [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10_Autism_Spectrum_Quotient",
        "Social_Responsiveness_Scale", "Age_Years", "Qchat_10_Score", "Speech Delay/Language Disorder",
        "Learning disorder", "Genetic_Disorders", "Depression",
        "Global developmental delay/intellectual disability", "Social/Behavioural Issues",
        "Childhood Autism Rating Scale", "Anxiety_disorder", "Sex", "Ethnicity", "Jaundice",
        "Family_mem_with_ASD", "Who_completed_the_test"
    ]
    
    # Ensure all features are included in the input
    processed_data = []
    for feature in expected_features:
        if feature in label_encoders:  # Encode categorical features
            value = data.get(feature, "Unknown")  # Default to "Unknown" if missing
            processed_data.append(label_encoders[feature].transform([value])[0])
        else:
            processed_data.append(data.get(feature, 0))  # Default numeric features to 0
    
    return np.array(processed_data).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    result = "Yes" if prediction[0] == 1 else "No"
    return jsonify({'ASD_traits': result})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)
