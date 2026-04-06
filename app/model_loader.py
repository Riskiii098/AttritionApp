import os
import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'employee_data_final.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.pkl')

_model = None
_features = None

def load_or_train_model():
    """ Mengambil data dari CSV jika belum ditraining, mentracking dengan MLflow dan menyimpan dengan joblib """
    global _model, _features
    if _model is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
            print("Memuat model dan fitur dari lokal file .pkl dengan joblib...")
            _model = joblib.load(MODEL_PATH)
            _features = joblib.load(FEATURES_PATH)
            print("Selesai! Model dimuat dari memory/file.")
        else:
            print("Initiating On-the-Fly Training and tracking with MLflow...")
            if not os.path.exists(CSV_PATH):
                raise FileNotFoundError(f"Data CSV tidak ditemukan di {CSV_PATH}")
                
            df = pd.read_csv(CSV_PATH)
            
            # Pisahkan kolom atribut X dan kolom target y
            X = df.drop(columns=['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], errors='ignore')
            y = df['Attrition']
            
            # Dataset Attrition sudah berformat Integer 0 (No) dan 1 (Yes)
            y_encoded = y.astype(int)
            
            # Encoding One-Hot untuk variabel teks kategori
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
            _features = X_encoded.columns.tolist()
            
            mlflow.set_experiment("Attrition_Prediction")
            with mlflow.start_run():
                _model = RandomForestClassifier(n_estimators=100, random_state=42)
                _model.fit(X_encoded, y_encoded)
                
                accuracy = _model.score(X_encoded, y_encoded)
                
                mlflow.log_param("n_estimators", 100)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.sklearn.log_model(_model, "model")
            
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(_model, MODEL_PATH)
            joblib.dump(_features, FEATURES_PATH)
            
            print("Selesai! Model berhasil dilatih, disimpan dengan joblib, dan dicatat di MLflow.")

def predict_from_dict(input_dict):
    """
    Memproses Dictionary dengan nama fitur asli, mengubahnya secara otomatis menjadi
    one-hot encoded array dan memunculkan hasil persentasi prediksi kemungkinan (Probability).
    """
    load_or_train_model()
    
    # Nilai default median (jaga-jaga untuk atribut yang ga diinput user)
    default_data = {
        'Age': 35, 'BusinessTravel': 'Travel_Rarely', 'DailyRate': 800,
        'Department': 'Research & Development', 'DistanceFromHome': 5,
        'Education': 3, 'EducationField': 'Life Sciences',
        'EnvironmentSatisfaction': 3, 'Gender': 'Male', 'HourlyRate': 65,
        'JobInvolvement': 3, 'JobLevel': 2, 'JobRole': 'Sales Executive',
        'JobSatisfaction': 3, 'MaritalStatus': 'Married', 'MonthlyIncome': 5000,
        'MonthlyRate': 15000, 'NumCompaniesWorked': 2, 'OverTime': 'No',
        'PercentSalaryHike': 15, 'PerformanceRating': 3, 'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1, 'TotalWorkingYears': 10, 'TrainingTimesLastYear': 3,
        'WorkLifeBalance': 3, 'YearsAtCompany': 5, 'YearsInCurrentRole': 3,
        'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 3
    }

    for key, value in input_dict.items():
        if key in default_data and str(value).strip() != "":
            if isinstance(default_data[key], int):
                default_data[key] = int(value)
            elif isinstance(default_data[key], float):
                default_data[key] = float(value)
            else:
                default_data[key] = str(value)

    df_user = pd.DataFrame([default_data])

    categorical_cols = df_user.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        df_user = pd.get_dummies(df_user, columns=categorical_cols)

    # Reindex mapping terhadap memori RAM Training (_features)
    df_user = df_user.reindex(columns=_features, fill_value=0)
    df_user = df_user.astype(float)

    # Prediksi
    prediction = _model.predict(df_user)
    probabilities = _model.predict_proba(df_user)[0]
    
    prob_attrition = probabilities[1]
    return int(prediction[0]), float(prob_attrition)

def predict(input_data):
    """ Backward compatibility jika input masih berupa list array 1 dimensi """
    load_or_train_model()
    data_2d = np.array(input_data).reshape(1, -1)
    
    prediction = _model.predict(data_2d)
    probabilities = _model.predict_proba(data_2d)[0]
    
    return int(prediction[0]), float(probabilities[1])
