from joblib import load
import joblib
import numpy as np
import lightgbm as lgb


model = joblib.load('lgbm_model.joblib')
scaler = joblib.load('scaler.joblib')
le_Sex = joblib.load('le_Sex.joblib')


'''input_data = [
    1,      # Sex (encoded)
    58.0,   # Age
    0,      # ChestPainType (encoded)
    120.0,  # RestingBP
    200.0,  # Cholesterol
    0,      # FastingBS
    1,      # RestingECG (encoded)
    160.0,  # MaxHR
    0,      # ExerciseAngina (encoded)
    1.5,    # Oldpeak
    2       # ST_Slope (encoded)
]'''
input_data=[54,1,1,150,195,0,1,122,0,0,1]

scaled_input = scaler.transform([input_data])
prediction = model.predict(scaled_input)
y_pred_class = [1 if x > 0.5 else 0 for x in prediction]  
print(prediction)
print("Prediction:", y_pred_class)  





