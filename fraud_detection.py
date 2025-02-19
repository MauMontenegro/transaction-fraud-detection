import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('transaction_data_with_features.csv')

# Define features and target
X = df.drop(columns=['Is_Fraud', 'Transaction_ID'])  # Features
y = df['Is_Fraud']  # Target variable

# Identify categorical and numerical columns
categorical_features = ['Manager_Seniority', 'Manager_Location', 'Transaction_Type', 'Transaction_Category']
numerical_features = ['Transaction_Amount', 'Authorization_Limit', 'Transaction_Frequency', 
                      'Customer_Risk_Score', 'Day_of_Week', 'Transaction_Hour', 'Manager_Transaction_History']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),  # Standardize numerical features
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
])

# Create a pipeline with preprocessing and model training
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Model evaluation
print(classification_report(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(pipeline, 'fraud_detection_model.pkl')