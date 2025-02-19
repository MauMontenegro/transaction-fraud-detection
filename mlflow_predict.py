import pandas as pd
import joblib
import mlflow
from datetime import datetime

# Initialize MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Local MLflow server

mlflow.set_experiment("Fraud_Detection_Batch")

with mlflow.start_run():
    # Log metadata
    mlflow.log_param("batch_run_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Load the best model
    model_path = "fraud_detection_model.pkl"
    model = joblib.load(model_path)

    # Load new transactions
    df = pd.read_csv("transaction_data_with_features.csv")

    # Predict fraud
    df["fraud_prediction"] = model.predict(df.drop(columns=["Transaction_ID"]))

    # Log predictions as an artifact
    output_file = f"fraud_report_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(output_file, index=False)

    mlflow.log_artifact(output_file)

    print(f"✅ Fraud report saved & logged in MLflow: {output_file}")
