import pandas as pd

df= pd.read_excel("C:/Users/T & A/Documents/AI FOR SOFTWARE ENGINEERING/Week 2/WIR.xlsx")

excel_path = ("C:/Users/T & A/Documents/AI FOR SOFTWARE ENGINEERING/Week 2/WIR.xlsx")
def clean_and_process_data(excel_path):
	df = pd.read_excel(excel_path)
	# Example cleaning: drop rows with any missing values
	df = df.dropna()
	return df

# Example usage:
excel_document_path = ("C:/Users/T & A/Documents/AI FOR SOFTWARE ENGINEERING/Week 2/WIR.xlsx")
cleaned_df = clean_and_process_data(excel_document_path)
print(cleaned_df.head())

df.isnull()
df.isnull().sum()  # Check for missing values in each column

print(df.isnull().sum() * 100 / len(df))
print(df.isnull().sum().sum())

df_filled_zero = df.fillna(15)
# Or to modify the DataFrame in place:
# df.fillna(0, inplace=True)

numerical_cols = df.select_dtypes(include=['number']).columns

for col in numerical_cols:
    if df[col].isnull().any(): # Check if the column actually has missing values
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"Filled missing values in column '{col}' with mean: {mean_value:.2f}")

        print(f"\nTotal missing values in DataFrame: {df.isnull().sum().sum()}")

import pandas as pd

# Assuming your DataFrame is named 'df'
# Print all column names
print(df.columns)

import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming your DataFrame is named 'df'
# And that you have already filled missing values

# Define the target variable (y) - use the integer 2016
target_year = 2016
y = df[target_year]

# Define the features (X)
# These will be the country/indicator identifiers and all years up to 2015
# Use integers for the year columns
year_columns_for_features = [year for year in range(1990, 2016)] # Years from 1990 up to 2015 as integers

# Columns that uniquely identify the observation but are not numerical features for prediction
id_columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']

# Combine ID columns with historical year columns to form your features X
X = df[id_columns + year_columns_for_features]

print("Shape of Features (X):", X.shape)
print("Shape of Target (y):", y.shape)

# Display the first few rows of X and y to confirm
print("\nFirst 5 rows of X:")
print(X.head())
print(f"\nFirst 5 rows of y ({target_year}):")
print(y.head())

# Convert categorical features into numerical format using One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], drop_first=True)

print("\nShape of X after One-Hot Encoding:", X_encoded.shape)
print("\nFirst 5 rows of X_encoded:")
print(X_encoded.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print(f"\nTraining set size (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Testing set size (X_test, y_test): {X_test.shape}, {y_test.shape}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Assuming your DataFrame 'df' is loaded and missing values are filled
# And target_year = 2016

# --- (Previous code for defining y and X) ---
target_year = 2016
y = df[target_year]

year_columns_for_features = [year for year in range(1990, 2016)]
id_columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
X = df[id_columns + year_columns_for_features]

# Convert categorical features into numerical format using One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], drop_first=True)

# *** ADD THIS LINE TO CONVERT ALL COLUMN NAMES TO STRINGS ***
X_encoded.columns = X_encoded.columns.astype(str)
print("\nColumn data types after converting to string:", X_encoded.columns.dtype) # Verify all are strings

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print(f"\nTraining set size (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Testing set size (X_test, y_test): {X_test.shape}, {y_test.shape}")

# --- (Model Training and Evaluation Code) ---
# Initialize the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Model
print("Training the model...")
model.fit(X_train, y_train) # This line should now work without the TypeError
print("Model training complete!")

# Make Predictions on the Test Set
y_pred = model.predict(X_test)
print("\nPredictions made on the test set.")

# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Optional: Display some actual vs. predicted values
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nSample of Actual vs. Predicted Values:")
print(results_df.head())
