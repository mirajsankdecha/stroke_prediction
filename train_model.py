# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('C:/Users/Lenovo/Downloads/stroke_prediction/data/healthcare-dataset-stroke-data.csv')
    
    # Handle missing BMI values
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    median_bmi = df['bmi'].median()
    df['bmi'].fillna(median_bmi, inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

def train_model():
    print("Loading and preprocessing data...")
    df, label_encoders = load_and_preprocess_data()
    
    # Separate features and target
    X = df.drop(['stroke'], axis=1)
    y = df['stroke']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Stroke cases: {y.sum()} out of {len(y)} ({y.mean()*100:.2f}%)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    with open('model/best_stroke_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save label encoders
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save metadata
    metadata = {
        'features': list(X.columns),
        'model_type': 'RandomForestClassifier',
        'training_date': pd.Timestamp.now().isoformat(),
        'dataset_size': len(df),
        'stroke_cases': int(y.sum())
    }
    
    with open('model/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Model saved successfully!")
    print("Files saved:")
    print("- model/best_stroke_model.pkl")
    print("- model/label_encoder.pkl") 
    print("- model/metadata.json")

if __name__ == "__main__":
    train_model()
