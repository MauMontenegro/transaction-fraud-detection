import joblib
from sklearn.metrics import classification_report
import pandas as pd
from datetime import datetime

# Load Model
loaded_model = joblib.load("fraud_detection_model.pkl")

# Load the dataset
df = pd.read_csv('transaction_data_with_features.csv')

# Model inference
df["fraud_prediction"] = loaded_model.predict(df.drop(columns=["Transaction_ID"]))

df["fraud_label"] = df["fraud_prediction"].apply(lambda x: "Fraud" if x == 1 else "Not Fraud")

# Generate a text report
report_filename = f"fraud_report_{datetime.now().strftime('%Y-%m-%d')}.txt"

# Save Report
with open(report_filename, "w") as report:
    report.write(" DAILY FRAUD DETECTION REPORT \n")
    report.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
    report.write("=" * 50 + "\n\n")

    for _, row in df.iterrows():
        report.write(f"Transaction ID: {row['Transaction_ID']}\n")
        report.write(f"Amount: ${row['Transaction_Amount']:.2f}\n")
        report.write(f"Prediction:  {row['fraud_label']} \n")
        report.write("-" * 50 + "\n\n")

print(f"✅ Fraud report generated: {report_filename}")



