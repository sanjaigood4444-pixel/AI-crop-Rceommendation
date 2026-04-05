# 🌾 AI Crop Recommendation & Yield Prediction System

## 📌 Description
This project predicts the most suitable crop and expected yield using soil nutrients and environmental conditions.

## ⚙️ Technologies Used
- Python
- Streamlit
- Machine Learning (Random Forest)

## 🚀 Features
- Crop Recommendation 🌾
- Yield Prediction 📈
- Interactive UI using Streamlit
- Image-based visualization

## 📊 Input Parameters
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
## Structure:
app.py
main.py
dataset.csv
images/

## 📊 Model Performance

The following machine learning models were implemented and evaluated:

| Model | Accuracy |
|------|----------|
| Random Forest | 99.54% |
| XGBoost | 98.64% |
| CatBoost | 99.32% |

### 🔍 Observations
- Random Forest achieved the highest accuracy.
- CatBoost performed competitively with strong results.
- XGBoost also provided reliable predictions.

### ✅ Conclusion
Random Forest was selected as the final model due to its superior performance and stability.

## 📸 Sample Output

### 🌾 Crop Prediction UI
<img width="870" src="https://github.com/user-attachments/assets/31d9f27b-ea20-412c-b3f4-c00bc2b768d7" />

### 📈 Prediction Result
<img width="804" src="https://github.com/user-attachments/assets/61e72baf-aaf8-4ee6-858d-a17a5957d992" />
