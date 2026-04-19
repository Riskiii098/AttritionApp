import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import dagshub
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
            
            # Bagi data: 80% untuk latihan, 20% untuk ujian
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
            
            # Cek environment untuk DagsHub
            dagshub_token = os.environ.get("DAGSHUB_TOKEN") or os.environ.get("DAGSHUB_USER_TOKEN")
            if dagshub_token:
                try:
                    # Hapus ketergantungan pada fungsi dagshub.init() yang bikin server hang!
                    # Gunakan native MLflow API yang murni HTTP request
                    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Riskiii098/AttritionApp.mlflow"
                    os.environ["MLFLOW_TRACKING_USERNAME"] = "Riskiii098"
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
                    
                    print("Initiating pure MLflow integration to DagsHub...")
                    mlflow.set_tracking_uri("https://dagshub.com/Riskiii098/AttritionApp.mlflow")
                    mlflow.set_experiment("Attrition_Prediction")
                    
                    # Membuat nama run otomatis dengan timestamp
                    run_name = f"RandomForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    with mlflow.start_run(run_name=run_name):
                        _model = train_logic(X_train, y_train, X_test, y_test)
                        mlflow.sklearn.log_model(_model, "model")
                except Exception as e:
                    print(f"Tracking error (DagsHub/MLflow): {e}. Lanjut training tanpa tracking.")
                    _model = train_logic(X_train, y_train, X_test, y_test)
            else:
                print("DAGSHUB_TOKEN tidak ditemukan. Melakukan training lokal tanpa tracking.")
                _model = train_logic(X_train, y_train, X_test, y_test)
            
            # Coba simpan model ke file, abaikan jika read-only permission (misal di Hugging Face)
            try:
                os.makedirs(MODEL_DIR, exist_ok=True)
                joblib.dump(_model, MODEL_PATH)
                joblib.dump(_features, FEATURES_PATH)
                print("Selesai! Model berhasil dilatih dan disimpan dengan joblib.")
                
                # Integrasi Auto-Upload ke Hugging Face Model Hub (Khusus Space jika token disetel)
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    from huggingface_hub import HfApi
                    api = HfApi(token=hf_token)
                    print("Mengirim salinan fisik model .pkl ke repositori Model Hugging Face Anda...")
                    api.upload_file(
                        path_or_fileobj=MODEL_PATH,
                        path_in_repo="rf_model.pkl",
                        repo_id="Riskiii/Attrition-Model"
                    )
                    api.upload_file(
                        path_or_fileobj=FEATURES_PATH,
                        path_in_repo="features.pkl",
                        repo_id="Riskiii/Attrition-Model"
                    )
                    print("Yeay! Model berhasil mendarat di https://huggingface.co/Riskiii/Attrition-Model")
                else:
                    print("Lewati auto-upload Model: Variabel HF_TOKEN tidak ada di environment.")
                    
            except Exception as e:
                print(f"Peringatan: Gagal menyimpan/mengupload model. Berjalan di RAM. {e}")

def train_logic(X_train, y_train, X_test, y_test):
    """ Logika inti training RandomForest agar bisa dipanggil dengan atau tanpa MLflow """
    # Menambahkan class_weight='balanced' agar model lebih sensitif terhadap karyawan yang resign
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Uji pada data yang BELUM pernah dilihat (X_test)
    # Menggunakan Threshold 0.40 untuk deteksi Risiko (Logika Bisnis yang Benar)
    probs = model.predict_proba(X_test)[:, 1]
    predictions = (probs >= 0.40).astype(int)
    
    # Hitung Metrik Lengkap pada Threshold 0.40
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f"Final Model Trained! (Threshold: 0.40)")
    print(f"Metrics -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
    
    # Log metrics ke console (akan muncul di log Hugging Face)
    if mlflow.active_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("threshold", 0.40)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
    return model

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
