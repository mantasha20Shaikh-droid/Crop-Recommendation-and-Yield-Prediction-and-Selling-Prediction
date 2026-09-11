AgriSmart Model Training Notes

Recommended authoritative flow

Use one clear training definition per prediction task.

1. Crop recommendation

Target:

label

Features:

N, P, K, temperature, humidity, ph, rainfall

Model:

RandomForestClassifier

Output artifact:

models/rf_recommendation.joblib

2. Yield prediction

Target:

Yield

Recommended features:

Crop
State
Season
Area
Annual_Rainfall
Fertilizer
Pesticide

Do not include Yield itself or other variables that directly reveal the target.

Output artifacts:

models/rf_yield.joblib
models/yield_features.joblib

3. Selling / price prediction

Recommended target:

Price per kg (₹)

Recommended features:

Crop
State
Season
Quantity (kg)

After prediction:

estimated_total_value = quantity_kg * predicted_price_per_kg

Output artifacts expected by the corrected Streamlit app:

models/rf_selling.joblib
models/selling_features.joblib

Important existing-project conflict

train_models.py currently trains the selling model with:

Price per kg (₹)

as an input feature while predicting:

Total Value (₹)

This is mathematically very close to giving the model the answer because:

Total Value ≈ Quantity × Price per kg

Therefore that training setup can produce an unrealistically high score.

train_selling.py is closer to the intended price-prediction design because it targets the price column.

Keep only one authoritative selling-model pipeline in the final project and retrain the saved rf_selling.joblib accordingly.

Retraining

After changing a dataset or feature definition:

python train_models.py

or use the dedicated selling pipeline as appropriate.

Then make sure the saved model's expected input columns exactly match the Streamlit application.

Evaluation

Report at least:

MAE
RMSE
R²

For classification, report:

Accuracy

Do not present a percentage as "accuracy" when the metric is actually R².