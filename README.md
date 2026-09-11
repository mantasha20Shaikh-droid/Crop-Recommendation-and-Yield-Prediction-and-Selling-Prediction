**CROP RECOMMENDATION & YIELD PREDICTION**

AgriSmart Intelligence Portal

AgriSmart is an AI-assisted agriculture application built with Streamlit, Scikit-Learn, Pandas, Plotly, and SQLite.

The project provides:

🌱 Crop Recommendation

📈 Crop Yield Prediction

💰 Selling / Price Forecast

📊 Market Price Trends

🗺️ Regional / APMC market visualization

👤 User registration and login

🔐 Persistent authentication tokens

🛠️ Admin user-management CRUD

📜 Prediction history / audit logs

🤖 AgriBot / assistant interface

🌦️ Weather forecast integration through Open-Meteo

🚀 Optional FastAPI backend

Project structure

Project_503_AIMDD/
│
├── Apps.py                         # Main Streamlit application
├── main.py                         # Optional FastAPI backend
├── database.py                     # FastAPI-side SQLite helpers
├── map.py                          # Separate market-map helper
│
├── train_models.py                 # Model training script
├── train_selling.py                # Pipeline-based selling-price training
├── evaluate_selling.py             # Selling-model evaluation utility
├── test_auth_persistence.py        # Authentication persistence test
│
├── Crop_Recommendation_Final.csv   # Crop recommendation dataset
├── Crop_Yield_Final.csv            # Yield dataset
├── Crop_Selling_Final.csv          # Selling/market dataset
│
├── models/
│   ├── rf_recommendation.joblib
│   ├── rf_yield.joblib
│   ├── rf_selling.joblib
│   ├── yield_features.joblib
│   └── selling_features.joblib
│
├── PIC/                            # Application screenshots/assets
├── Structure/                      # College/project documentation templates
├── Bg.png, Image*.png, rice.webp   # UI assets
│
├── agrismart.db                    # Streamlit application database
├── crop_database.db                # FastAPI database
└── users.db                        # Existing project database file

Main entry point

The main application is:

streamlit run Apps.py

Open the local Streamlit URL shown in the terminal.

For the optional FastAPI backend:

uvicorn main:app --reload

API documentation will be available from FastAPI's interactive docs.

Installation

1. Create a virtual environment

Windows:

py -3.12 -m venv .venv
.venv\Scripts\activate

Linux/macOS:

python3 -m venv .venv
source .venv/bin/activate

2. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

3. Run AgriSmart

streamlit run Apps.py

4. Optional: run the FastAPI backend

From the project directory:

uvicorn main:app --reload

Machine-learning models

Crop Recommendation

Input features:

N
P
K
temperature
humidity
ph
rainfall

Target:

label

The application uses a Random Forest classifier.

Crop Yield Prediction

The intended target is:

Yield

The feature set used by the corrected application is:

Crop
State
Season
Area
Annual_Rainfall
Fertilizer
Pesticide

Important: Production should not be used as an input when predicting Yield when the dataset relationship can expose the target. Using target-derived information can cause data leakage and unrealistic evaluation.

Selling / Price Forecast

The corrected application treats:

Price per kg (₹)

as the prediction target and calculates estimated total value separately:

Total Value = Quantity (kg) × Predicted Price per kg

Important model-file warning

The project currently contains two different selling-model training approaches:

train_models.py trains rf_selling.joblib to predict Total Value (₹) while also using Price per kg (₹) as an input feature.

train_selling.py trains a separate pipeline intended to predict the price column.

These approaches are not equivalent.

For the current corrected Streamlit application, the recommended approach is:

Target      : Price per kg (₹)
Inputs      : Crop, State, Season, Quantity (kg)
Total value : calculated after prediction

Regenerate the selling model before deployment so that the saved rf_selling.joblib matches the application.

Datasets

The project contains three CSV datasets.

Crop recommendation

Crop_Recommendation_Final.csv

Columns:

N, P, K, temperature, humidity, ph, rainfall, label

Yield

Crop_Yield_Final.csv

Columns:

Crop, Crop_Year, Season, State, Area, Production,
Annual_Rainfall, Fertilizer, Pesticide, Yield

Selling

Crop_Selling_Final.csv

Columns:

Crop, State, Season, Quantity (kg),
Price per kg (₹), Total Value (₹)

Database design

The Streamlit app uses:

agrismart.db

It stores users, authentication tokens, and prediction records.

The FastAPI backend uses:

crop_database.db

for users and API history.

These are currently separate database systems. They should not be assumed to be synchronized.

User-management CRUD

The admin area supports:

CREATE → Create a user
READ   → View registered users
UPDATE → Edit user details
DELETE → Permanently delete a user

The full delete workflow is intended to remove the selected user's:

account record

authentication tokens

prediction history

and then refresh the admin tables.

Security notes

This is primarily a college/project application and should be hardened before production use.

Before public deployment:

Use strong, unique administrator credentials.

Do not commit real passwords or database files to Git.

Use environment variables or a secret manager for admin credentials.

Restrict FastAPI CORS instead of allowing all origins.

Use a stronger password hashing algorithm such as Argon2 or bcrypt for production.

Use HTTPS.

Add proper authorization checks to all admin API operations.

Avoid exposing arbitrary SQL execution over an internet-facing API.

Back up the SQLite databases before destructive CRUD operations.

External service

Weather data is requested from:

Open-Meteo API

The application also references Google Fonts from CSS.

Internet access is therefore required for live weather data and remotely loaded fonts. The core ML predictions can operate from local files/models.

Important files for submission

For a college submission, these files are normally important:

Apps.py
requirements.txt
README.md
LICENSE
.gitignore
Crop_Recommendation_Final.csv
Crop_Yield_Final.csv
Crop_Selling_Final.csv
models/
PIC/
Structure/

The SQLite database files may be kept for a demo with existing accounts/data, but for a clean source-code submission it is safer to initialize fresh databases.

Troubleshooting

ModuleNotFoundError

Activate the virtual environment and run:

pip install -r requirements.txt

Model file not found

Check:

models/rf_recommendation.joblib
models/rf_yield.joblib
models/rf_selling.joblib

Dataset not found

Run Streamlit from the project folder, or use the corrected application that resolves project-relative paths.

UNIQUE constraint failed: users.username

This means the username already exists in the SQLite database. The corrected registration/admin CRUD logic checks usernames before insertion.

FastAPI cannot load a model

Start uvicorn from the project directory so the models/ paths resolve correctly, or update main.py to use paths based on __file__.

License

This project includes a project-level MIT license in LICENSE.

Important: the project license does not automatically grant rights to third-party datasets, images, fonts, or external services. Verify the original license/usage terms for those materials separately.
