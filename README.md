import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
data = pd.read_csv('house_prices_dataset.csv')

# Step 2: Separate features (X) and target variable (y)
X = data.drop('SalePrice', axis=1)  # Features
y = data['SalePrice']  # Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Step 5: Define preprocessing steps for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
    ('scaler', StandardScaler())  # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute missing values with 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Step 6: Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 7: Preprocess the training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Step 8: Initialize the model (Linear Regression in this case)
model = LinearRegression()

# Step 9: Train the model
model.fit(X_train_processed, y_train)

# Step 10: Predict on the test set
y_pred = model.predict(X_test_processed)

# Step 11: Evaluate model performance (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Step 12: Example of fine-tuning (e.g., Ridge Regression with regularization)
from sklearn.linear_model import Ridge

model_ridge = Ridge(alpha=0.1)  # Example of Ridge regression with regularization parameter alpha
model_ridge.fit(X_train_processed, y_train)
y_pred_ridge = model_ridge.predict(X_test_processed)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f'Ridge Regression RMSE: {rmse_ridge:.2f}')
