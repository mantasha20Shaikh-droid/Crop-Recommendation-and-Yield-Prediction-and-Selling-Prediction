# Crop Recommendation and Yield Prediction and Selling Prediction Streamlit Application

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

ROOT = os.path.dirname(__file__)
CSV = os.path.join(ROOT, 'Crop_Selling_Final.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_OUT = os.path.join(MODEL_DIR, 'rf_selling_pipeline.joblib')
FEATURES_OUT = os.path.join(MODEL_DIR, 'selling_features.joblib')

if not os.path.exists(CSV):
    print('ERROR: CSV not found at', CSV)
    raise SystemExit(2)

df = pd.read_csv(CSV)

# target
price_col = next((c for c in df.columns if 'price' in c.lower()), None)
if price_col is None:
    print('ERROR: price column not found')
    raise SystemExit(2)

# choose features
cat_cols = [c for c in ['Crop', 'State', 'Season'] if c in df.columns]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != price_col]

print('Categorical cols:', cat_cols)
print('Numeric cols:', num_cols)

X = df[cat_cols + num_cols].copy()
y = df[price_col].astype(float).values

# simple cleaning
X[ num_cols ] = X[num_cols].fillna(0)
X[ cat_cols ] = X[cat_cols].fillna('Unknown')

# preprocess
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('num', StandardScaler(), num_cols)
])

pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training on', len(X_train), 'samples...')
pipeline.fit(X_train, y_train)
print('Training complete.')

# evaluate
y_pred = pipeline.predict(X_test)
errors = np.abs(y_pred - y_test)
within_5 = (errors <= 5).sum()
within_10 = (errors <= 10).sum()
within_20 = (errors <= 20).sum()

n = len(y_test)
print(f'Test samples: {n}')
print(f'Within ₹5: {within_5}/{n} = {within_5/n*100:.2f}%')
print(f'Within ₹10: {within_10}/{n} = {within_10/n*100:.2f}%')
print(f'Within ₹20: {within_20}/{n} = {within_20/n*100:.2f}%')
print('MAE:', mean_absolute_error(y_test, y_pred))

# save pipeline and feature names
joblib.dump(pipeline, MODEL_OUT)

# derive feature names for saved pipeline
try:
    ohe = pipeline.named_steps['pre'].named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(cat_cols).tolist()
except Exception:
    cat_names = []
num_names = num_cols
feature_names = cat_names + num_names
joblib.dump(feature_names, FEATURES_OUT)
print('Saved pipeline to', MODEL_OUT)
print('Saved feature list to', FEATURES_OUT)
