import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import pickle

# Load and preprocess the dataset
def preprocess_data():
    data = pd.read_csv('data.csv')
    
    # Drop irrelevant columns
    data.drop(columns=["CASE_NO_PATIENT'S"], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column != 'ASD_traits':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le
    
    # Separate features and target
    X = data.drop(columns=['ASD_traits'])
    y = LabelEncoder().fit_transform(data['ASD_traits'])
    
    return X, y, label_encoders

# Train and save the model
def train_model():
    X, y, label_encoders = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Save the model and encoders
    with open('model.pkl', 'wb') as model_file:
        pickle.dump((model, label_encoders), model_file)
    print("Model trained and saved!")

if __name__ == '__main__':
    train_model()
