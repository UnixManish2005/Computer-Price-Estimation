from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import pandas as pd

# Load the dataset
##file_path = "C:\Users\my\Desktop\ML Asssmnt\ComputerPrice.py"
df = pd.read_csv("CPE.csv")

# Display basic information and first few rows
df.info(), df.head()

# Drop unnecessary column
df = df.drop(columns=["Unnamed: 0"])

# Drop unnecessary columns (if any)
df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# Convert categorical variables to numerical
categorical_columns = ["cd", "multi", "premium"]
for col in categorical_columns:
    df[col] = df[col].map({"yes": 1, "no": 0})

# Define features and target variable
X = df.drop(columns=["price"])  # Features
y = df["price"]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")


# Function to predict price for new data
def predict_price(new_data):
    new_df = pd.DataFrame([new_data])
    for col in categorical_columns:
        new_df[col] = new_df[col].map({"yes": 1, "no": 0})
    return model.predict(new_df)[0]

# Example Prediction
sample_data = {"speed": 10, "hd": 100, "ram": 4, "screen": 10, "cd": "yes", "multi": "no", "premium": "no", "ads": 50, "trend": 1}
predicted_price = predict_price(sample_data)
print(f"Predicted Price: ${predicted_price:.2f}")