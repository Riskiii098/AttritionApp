import os
import sys
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Menambahkan path absolut agar gunicorn selalu menemukan model_loader tanpa Error ModuleNotFound
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import fungsi prediksi yang telah berevolusi memberikan return Tuple
from model_loader import predict, predict_from_dict

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(base_dir, 'frontend')

app = Flask(
    __name__,
    template_folder=os.path.join(frontend_dir, 'templates'),
    static_folder=os.path.join(frontend_dir, 'static')
)
CORS(app)  # Izinkan frontend/Looker eksternal menarik API predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        req_data = request.get_json()
        
        if not req_data or 'features' not in req_data:
            return jsonify({
                'status': 'error',
                'message': 'Input tidak valid. Payload JSON harus memuat key "features".'
            }), 400
        
        input_data = req_data['features']
        
        # Eksekusi adaptif! Menangkap tuple result berisi Kelas dan Probabilitas
        if isinstance(input_data, dict):
            prediction_result, prob_attrition = predict_from_dict(input_data)
        elif isinstance(input_data, list):
            prediction_result, prob_attrition = predict(input_data)
        else:
            raise ValueError("Key 'features' harus format array [1, 2...] atau JSON Keys {'Age': 35}!")
        
        # Logika Bisnis Akhir: Threshold 0.40 (40% Probabilitas sudah dianggap berisiko)
        # Sesuai dengan standar deteksi dini untuk mitigasi HR.
        is_attrition = 1 if prob_attrition >= 0.40 else 0
        
        # Ubah probabilitas decimal 0.0-1.0 menjadi nilai Persen XX.XX%
        prob_percentage = round(prob_attrition * 100, 2)

        if is_attrition == 1:
            human_readable_status = "High Risk (Perlu Perhatian Khusus)"
        else:
            human_readable_status = "Safe / Low Risk (Diprediksi Bertahan)"

        return jsonify({
            'status': 'success',
            'prediction': is_attrition,
            'prediction_text': human_readable_status,
            'probability_percentage': prob_percentage
        }), 200

    except Exception as e:
        error_message = f"Terjadi kesalahan saat memproses data: {str(e)}"
        print(error_message)
        traceback.print_exc()

        return jsonify({
            'status': 'error',
            'message': error_message
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
