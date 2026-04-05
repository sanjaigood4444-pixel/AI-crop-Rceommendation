# ================================
# STEP 1: Import Libraries
# ================================
import pandas as pd

# ================================
# STEP 2: Load Dataset
# ================================
data = pd.read_csv("dataset.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# ================================
# STEP 3: Preprocessing
# ================================
# Only 'label' (crop) is categorical

from sklearn.preprocessing import LabelEncoder

le_crop = LabelEncoder()

data['label'] = le_crop.fit_transform(data['label'])

# ================================
# STEP 4: Split Dataset
# ================================
from sklearn.model_selection import train_test_split

# Features (X) and Target (y)
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ================================
# STEP 5: Train Models
# ================================

# ---- Random Forest ----
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# ---- XGBoost ----
from xgboost import XGBClassifier

xgb_model = XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))


# ---- CatBoost ----
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(X_train, y_train)

y_pred_cat = cat_model.predict(X_test)

print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_cat))


# ================================
# STEP 5.1: Cross Validation
# ================================
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X, y, cv=5)

print("\nCross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

# ================================
# STEP 6: Yield Prediction (REAL DATA)
# ================================

# Define features and target
X_reg = data.drop(['label', 'Yield'], axis=1)
y_reg = data['Yield']

# Split data
from sklearn.model_selection import train_test_split

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train model
from sklearn.ensemble import RandomForestRegressor

reg_model = RandomForestRegressor()
reg_model.fit(X_train_r, y_train_r)

# Predict
y_pred_r = reg_model.predict(X_test_r)

# Evaluate
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

print("\nYield Prediction RMSE:", rmse)