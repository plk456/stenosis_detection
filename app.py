from flask import Flask, render_template, request,redirect , jsonify
import joblib  
from sklearn.preprocessing import StandardScaler,LabelEncoder
from ultralytics import YOLO
import yaml
import os
import uuid
import numpy as np
import pydicom
from werkzeug.utils import secure_filename
import cv2
import base64
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)

model = joblib.load('lgbm_model.joblib')


@app.route('/')
def fun():
    return render_template('detail.html')



@app.route('/heartdisease',methods=['GET','POST'])
def home():
    if request.method=='POST':
        return render_template('heartdisease.html')


model = joblib.load('lgbm_model.joblib')  # Replace with your actual model file
le_Sex = joblib.load('le_Sex.joblib')
le_ChestPainType = joblib.load('le_ChestPainType.joblib')
le_RestingECG = joblib.load('le_RestingECG.joblib')
le_ExerciseAngina = joblib.load('le_ExerciseAngina.joblib')
le_ST_Slope = joblib.load('le_ST_Slope.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    try:
        # Get form data
        form_data = {
            'sex': request.form.get('sex'),
            'age': float(request.form.get('age')),
            'chest_pain_type': request.form.get('chest_pain_type'),
            'resting_bp': float(request.form.get('resting_bp')),
            'cholesterol': float(request.form.get('cholesterol')),
            'fasting_bs': float(request.form.get('fasting_bs')),
            'resting_ecg': request.form.get('resting_ecg'),
            'max_hr': float(request.form.get('max_hr')),
            'exercise_angina': request.form.get('exercise_angina'),
            'oldpeak': float(request.form.get('oldpeak')),
            'st_slope': request.form.get('st_slope')
        }

        # Encode categorical variables
        processed_data = [
            le_Sex.transform([form_data['sex']])[0] if form_data['sex'] in le_Sex.classes_ else -1,
            form_data['age'],
            le_ChestPainType.transform([form_data['chest_pain_type']])[0] if form_data['chest_pain_type'] in le_ChestPainType.classes_ else -1,
            form_data['resting_bp'],
            form_data['cholesterol'],
            form_data['fasting_bs'],
            le_RestingECG.transform([form_data['resting_ecg']])[0] if form_data['resting_ecg'] in le_RestingECG.classes_ else -1,
            form_data['max_hr'],
            le_ExerciseAngina.transform([form_data['exercise_angina']])[0] if form_data['exercise_angina'] in le_ExerciseAngina.classes_ else -1,
            form_data['oldpeak'],
            le_ST_Slope.transform([form_data['st_slope']])[0] if form_data['st_slope'] in le_ST_Slope.classes_ else -1
        ]

        # Standardize the input data
        input_data_scaled = scaler.transform([processed_data])

        # Make prediction (assuming your model outputs 1 for high risk, 0 for low risk)
        prediction = model.predict(input_data_scaled)
        y_pred_class = [1 if x > 0.5 else 0 for x in prediction]  
        risk_score = float(prediction[0])
        risk_level = 'high_risk' if y_pred_class[0] == 1 else 'low_risk'

        # Generate recommendations based on risk level
        if risk_level == 'high_risk':
            recommendations = [
                "Schedule an appointment with your doctor immediately",
                "Monitor your blood pressure daily",
                "Adopt a low-sodium, heart-healthy diet",
                "Begin a supervised exercise program",
                "If you smoke, seek help to quit immediately",
                "Limit alcohol consumption"
            ]
        else:
            recommendations = [
                "Continue regular health check-ups",
                "Maintain a balanced diet with plenty of fruits and vegetables",
                "Engage in 150 minutes of moderate exercise weekly",
                "Monitor your cholesterol levels annually",
                "Practice stress-reduction techniques",
                "Avoid tobacco products"
            ]

        return jsonify({
            'success': True,
            'risk_level': risk_level,
            'risk_score': risk_score, 
            'form_data': form_data,
            'recommendations': recommendations
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Invalid input value. Please check your form data.'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing your request. Please try again.'
        }), 500



@app.route('/classificationpred', methods=['GET', 'POST'])
def classpred():
    if request.method == 'POST':
        # Handle any POST data if needed
        pass
    return render_template('classificationpred.html')



model2 = YOLO('my_model.pt')

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        print("\n=== NEW REQUEST ===")
        print("Request files:", request.files)
        
        if 'image' not in request.files:
            print("No 'image' key in request")
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files['image']
        print("Received file:", image_file.filename)

        # Validate filename
        if image_file.filename == '':
            print("Empty filename")
            return jsonify({"error": "No selected file"}), 400

        # Validate file extension
        if not allowed_file(image_file.filename):
            print("Invalid file extension")
            return jsonify({"error": "Invalid file type"}), 400

        # Create upload directory if missing
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")

        # Save file
        temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(image_file.filename))
        print("Saving to:", temp_path)
        image_file.save(temp_path)
        print("File saved successfully")

        # Verify file exists
        if not os.path.exists(temp_path):
            print("ERROR: File not saved correctly")
            return jsonify({"error": "File processing failed"}), 500

        # Model prediction
        print("Running prediction...")
        results = model2.predict(temp_path, conf=0.25, device='cpu')
        print("Prediction completed:", results)

        # Read the original image
        original_image = cv2.imread(temp_path)
        
        # Plot the results on the image
        annotated_image = results[0].plot()  # This adds bounding boxes and labels
        
        # Convert annotated image to base64
        detected_image_base64 = encode_image_to_base64(annotated_image)

        if not results or len(results[0].boxes) == 0:
            print("No detections found")
            return jsonify({
                "condition": "healthy", 
                "confidence": 0,
                "detected_image": detected_image_base64  # Still return the original image
            })

        # Process results
        best_result = results[0].boxes[0]
        class_name = model2.names[int(best_result.cls)]
        confidence = float(best_result.conf) * 100
        print(f"Detection: {class_name} ({confidence:.2f}%)")

        return jsonify({
            "condition": class_name,
            "confidence": round(confidence, 2),
            "detected_image": detected_image_base64
        })

    except Exception as e:
        print("\n!!! ERROR !!!")
        print(str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

    finally:
        # Clean up
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            print("Temp file cleaned up")


@app.route('/cancerprediction', methods=['GET', 'POST'])
def cancer_prediction_page():
    if request.method == 'POST':
        # Handle form submission if needed
        pass
    return render_template('oncology_analysis.html')


model3 = YOLO('my_model3.pt')
# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}  # Added dcm for DICOM images
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file2(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image2_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/api/analyze-oncology', methods=['POST'])
def analyze_oncology():
    try:
        print("\n=== NEW ONCOLOGY ANALYSIS REQUEST ===")
        print("Request files:", request.files)
        
        if 'image' not in request.files:
            print("No 'image' key in request")
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files['image']
        print("Received file:", image_file.filename)

        # Validate filename
        if image_file.filename == '':
            print("Empty filename")
            return jsonify({"error": "No selected file"}), 400

        # Validate file extension
        if not allowed_file(image_file.filename):
            print("Invalid file extension")
            return jsonify({"error": "Invalid file type. Supported types: PNG, JPG, JPEG, DICOM"}), 400

        # Create upload directory if missing
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")

        # Save file
        temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(image_file.filename))
        print("Saving to:", temp_path)
        image_file.save(temp_path)
        print("File saved successfully")

        # Verify file exists
        if not os.path.exists(temp_path):
            print("ERROR: File not saved correctly")
            return jsonify({"error": "File processing failed"}), 500

        # Model prediction
        print("Running oncology prediction...")
        results = model3.predict(temp_path, conf=0.25, device='cpu')
        print("Prediction completed:", results)

        # Read the original image
        original_image = cv2.imread(temp_path)
        
        # Plot the results on the image
        annotated_image = results[0].plot()  # This adds bounding boxes and labels
        
        # Convert annotated image to base64
        detected_image_base64 = encode_image_to_base64(annotated_image)

        if not results or len(results[0].boxes) == 0:
            print("No cancer detections found")
            return jsonify({
                "condition": "No malignant findings detected", 
                "confidence": 0,
                "detected_image": detected_image_base64
            })

        # Process results
        best_result = results[0].boxes[0]
        class_name = model3.names[int(best_result.cls)]
        confidence = float(best_result.conf) * 100
        print(f"Detection: {class_name} ({confidence:.2f}%)")

        return jsonify({
            "condition": class_name,
            "confidence": round(confidence, 2),
            "detected_image": detected_image_base64
        })

    except Exception as e:
        print("\n!!! ONCOLOGY ANALYSIS ERROR !!!")
        print(str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Oncology analysis failed", "details": str(e)}), 500

    finally:
        # Clean up
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            print("Temp file cleaned up")

if __name__ == '__main__':
    print("plk--->STARTING THE SERVER<---plk")
    app.run(host='0.0.0.0', port=5000,debug=True)
    
