# Crop Recommendation and Yield Prediction and Selling Prediction Streamlit Application

import os
import sys
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'rf_selling.joblib')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'models', 'selling_features.joblib')
CSV_PATH = os.path.join(os.path.dirname(__file__), 'Crop_Selling_Final.csv')

if not os.path.exists(MODEL_PATH):
    print('ERROR: model not found at', MODEL_PATH)
    sys.exit(2)
if not os.path.exists(CSV_PATH):
    print('ERROR: CSV not found at', CSV_PATH)
    sys.exit(2)

print('Loading model...')
model = joblib.load(MODEL_PATH)
print('Model loaded.')

print('Loading data...')
df = pd.read_csv(CSV_PATH)
print('Rows in data:', len(df))

# Identify target
possible_targets = [col for col in df.columns if 'price' in col.lower() or 'price per' in col.lower() or 'price' in col.lower()]
if not possible_targets:
    print('ERROR: Could not locate price column in CSV. Columns:', df.columns.tolist())
    sys.exit(2)
# pick first matching
price_col = possible_targets[0]
print('Using target column:', price_col)

# Load feature list if available
if os.path.exists(FEATURES_PATH):
    try:
        features = joblib.load(FEATURES_PATH)
        print('Using saved feature list with', len(features), 'features')
    except Exception as e:
        print('Could not load feature list:', e)
        features = None
else:
    features = None

# Prepare X
if features:
    missing = [c for c in features if c not in df.columns]
    if missing:
        print('WARNING: some expected features missing from CSV:', missing)
    X = df.reindex(columns=features).fillna(0)
else:
    # heuristic: use numeric columns except target
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if price_col in numeric:
        numeric.remove(price_col)
    X = df[numeric].fillna(0)
    print('Using numeric features:', X.columns.tolist())

y_true = df[price_col].astype(float).values

print('Predicting...')
try:
    y_pred = model.predict(X)
except Exception as e:
    print('ERROR during predict:', e)
    sys.exit(2)

errors = np.abs(y_pred - y_true)
within_10 = (errors <= 10).sum()
within_5 = (errors <= 5).sum()
within_20 = (errors <= 20).sum()

n = len(y_true)
print(f'Total samples: {n}')
print(f'Within ₹5: {within_5}/{n} = {within_5/n*100:.2f}%')
print(f'Within ₹10: {within_10}/{n} = {within_10/n*100:.2f}%')
print(f'Within ₹20: {within_20}/{n} = {within_20/n*100:.2f}%')
print('MAE:', np.mean(errors))
print('RMSE:', np.sqrt(np.mean(errors**2)))

# Output summary line for parsing
print('\nEVAL_SUMMARY:', within_10, n, f'{within_10/n*100:.4f}')
