import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set a random seed for reproducibility
np.random.seed(100)

# Define parameters
num_transactions = 1000         # Generate 1000 registers
manager_ids = [1, 2, 3, 4, 5]   # List of manager IDs
auth_limit_lower = 10           # Minimum limit
auth_limit_upper = 9999         # Maximum limit

# Create random date range for transaction times (within last year)
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(num_transactions)]

# Generate random transaction data
manager_ids_array = np.random.choice(manager_ids, num_transactions)
transaction_amounts = np.random.randint(5, 15000, num_transactions)  # Random amounts between 5 and 15000

# Fraud condition (manager authorizes a transaction above their limit)
fraud_condition = transaction_amounts > np.array([np.random.randint(auth_limit_lower, auth_limit_upper) 
                                                 for _ in range(num_transactions)])

# Generate additional features
transaction_times = [random.choice(date_range) for _ in range(num_transactions)]
manager_seniority = np.random.choice(['Junior', 'Mid', 'Senior'], num_transactions)  # Manager seniority
transaction_frequencies = np.random.randint(1, 10, num_transactions)  # Random transaction frequency for managers
locations = np.random.choice(['North', 'South', 'East', 'West'], num_transactions)  # Manager locations
customer_risk_score = np.random.randint(1, 101, num_transactions)  # Random risk score from 1 to 100
transaction_types = np.random.choice(['Purchase', 'Refund', 'Transfer', 'Withdrawal'], num_transactions)  # Transaction type
days_of_week = [transaction_time.weekday() for transaction_time in transaction_times]  # Day of the week (0=Monday, 6=Sunday)
transaction_hours = [transaction_time.hour for transaction_time in transaction_times]  # Hour of the transaction
manager_transaction_history = np.random.randint(1000, 50000, num_transactions)  # Total amount transacted by manager
transaction_categories = np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Furniture'], num_transactions)  # Transaction category

# Create the DataFrame from transaction
data = {
    'Transaction_ID': range(1, num_transactions + 1),
    'Manager_ID': manager_ids_array,
    'Transaction_Amount': transaction_amounts,
    'Authorization_Limit': [np.random.randint(auth_limit_lower, auth_limit_upper) for _ in range(num_transactions)],
    'Is_Fraud': fraud_condition,
    'Transaction_Time': transaction_times,
    'Manager_Seniority': manager_seniority,
    'Transaction_Frequency': transaction_frequencies,
    'Manager_Location': locations,
    'Customer_Risk_Score': customer_risk_score,
    'Transaction_Type': transaction_types,
    'Day_of_Week': days_of_week,
    'Transaction_Hour': transaction_hours,
    'Manager_Transaction_History': manager_transaction_history,
    'Transaction_Category': transaction_categories
}

# Create the pandas DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Optionally, save to a CSV file
df.to_csv('transactions.csv', index=False)
