import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# 1. Folders Setup
os.makedirs('dataset', exist_ok=True)
os.makedirs('model', exist_ok=True)

# 2. Data Generation
brands = [
    'BMW',
    'Mercedes',
    'Audi',
    'Jaguar',
    'Land Rover',
    'Maruti',
    'Hyundai',
    'Honda',
    'Toyota',
    'Ford',
]
fuel_types = ['Petrol', 'Diesel', 'CNG']
transmissions = ['Manual', 'Automatic']

np.random.seed(42)
n_samples = 1200

data = {
    'car_brand': np.random.choice(brands, n_samples),
    'car_age': np.random.randint(1, 15, n_samples),
    'kilometers_driven': np.random.randint(5000, 150000, n_samples),
    'engine_capacity': np.random.randint(800, 4000, n_samples),
    'mileage': np.round(np.random.uniform(8.0, 25.0, n_samples), 2),
    'owners': np.random.randint(1, 5, n_samples),
    'fuel_type': np.random.choice(fuel_types, n_samples),
    'transmission': np.random.choice(transmissions, n_samples),
}

df = pd.DataFrame(data)

# Pricing Logic
brand_bonus = {
    'Maruti': 100000,
    'Hyundai': 200000,
    'Honda': 350000,
    'Toyota': 500000,
    'Ford': 300000,
    'BMW': 2500000,
    'Mercedes': 2700000,
    'Audi': 2400000,
    'Jaguar': 2600000,
    'Land Rover': 3000000,
}
df['brand_val'] = df['car_brand'].map(brand_bonus)

df['selling_price'] = (
    500000
    + df['brand_val']
    - (df['car_age'] * 80000)
    - (df['kilometers_driven'] * 3)
    + (df['engine_capacity'] * 300)
    + (df['mileage'] * 5000)
    - (df['owners'] * 50000)
    + np.where(df['fuel_type'] == 'Diesel', 150000, 0)
    + np.where(df['transmission'] == 'Automatic', 200000, 0)
    + np.random.normal(0, 20000, n_samples)
)

df['selling_price'] = np.round(df['selling_price'], 2)
df.drop(columns=['brand_val'], inplace=True)

# Save CSV
df.to_csv('dataset/used_car_data.csv', index=False)

# 3. Encoding & Train/Test Split BEFORE Encoding to prevent Warning
X = df.drop(columns=['selling_price'])
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_cols = ['car_brand', 'fuel_type', 'transmission']
numeric_cols = [
    'car_age',
    'kilometers_driven',
    'engine_capacity',
    'mileage',
    'owners',
]

encoder = OneHotEncoder(
    sparse_output=False, handle_unknown='ignore', drop=None
)

# Fit encoder ONLY on training data
encoded_train = encoder.fit_transform(X_train[categorical_cols])
encoded_train_df = pd.DataFrame(
    encoded_train, columns=encoder.get_feature_names_out(categorical_cols)
)

encoded_test = encoder.transform(X_test[categorical_cols])
encoded_test_df = pd.DataFrame(
    encoded_test, columns=encoder.get_feature_names_out(categorical_cols)
)

X_train_final = pd.concat(
    [X_train[numeric_cols].reset_index(drop=True), encoded_train_df], axis=1
)
X_test_final = pd.concat(
    [X_test[numeric_cols].reset_index(drop=True), encoded_test_df], axis=1
)

# 4. Model Training
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_final, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test_final)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('--- Model Evaluation Metrics ---')
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.4f}')

# 6. Save Artifacts
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(model, 'model/car_price_model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')

print("Model trained and saved successfully as 'car_price_model.pkl'")