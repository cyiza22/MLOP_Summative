"""Flask API for ML model serving"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import tensorflow as tf
from werkzeug.utils import secure_filename

from src.preprocessing import preprocess_image
from src.prediction import ModelPredictor
from src.model import retrain_model

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = 'models/cifar10_classifier.h5'
UPLOAD_FOLDER = 'data/uploads'
TRAIN_FOLDER = 'data/train'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize predictor
try:
    predictor = ModelPredictor(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except:
    predictor = None
    print("⚠️ Model not found. Please train the model first.")

# Metrics
metrics = {
    'total_requests': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'start_time': datetime.now()
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve main UI"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    uptime = (datetime.now() - metrics['start_time']).total_seconds()
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': uptime,
        'model_loaded': predictor is not None,
        'total_requests': metrics['total_requests']
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Single image prediction"""
    metrics['total_requests'] += 1
    
    if predictor is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    try:
        img = Image.open(file.stream).convert('RGB')
        img_array = np.array(img)
        processed_img = preprocess_image(img_array)
        
        result = predictor.predict(processed_img)
        metrics['successful_predictions'] += 1
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        metrics['failed_predictions'] += 1
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """Upload new training data"""
    try:
        if 'images' not in request.files:
            return jsonify({'success': False, 'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        saved_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(TRAIN_FOLDER, filename)
                file.save(filepath)
                saved_files.append(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Uploaded {len(saved_files)} images',
            'files': saved_files
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining"""
    global predictor
    
    try:
        train_files = [f for f in os.listdir(TRAIN_FOLDER) if allowed_file(f)]
        
        if not train_files:
            return jsonify({'success': False, 'error': 'No training data'}), 400
        
        # Load training images
        X_new = []
        y_new = []
        
        for filename in train_files:
            filepath = os.path.join(TRAIN_FOLDER, filename)
            img = preprocess_image(filepath)
            X_new.append(img)
            # Random label for demo (in production, get from filename or form)
            y_new.append(np.random.randint(0, 10))
        
        X_new = np.array(X_new)
        y_new = tf.keras.utils.to_categorical(y_new, 10)
        
        # Validation data (small subset of CIFAR-10)
        (_, _), (X_val, y_val) = tf.keras.datasets.cifar10.load_data()
        X_val = X_val[:1000].astype('float32') / 255.0
        y_val = tf.keras.utils.to_categorical(y_val[:1000], 10)
        
        # Retrain
        new_model, history, new_model_path = retrain_model(
            MODEL_PATH, X_new, y_new, X_val, y_val,
            epochs=5, batch_size=32
        )
        
        # Update predictor
        predictor = ModelPredictor(new_model_path)
        
        final_acc = history.history['val_accuracy'][-1]
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'new_model_path': new_model_path,
            'training_samples': len(X_new),
            'final_accuracy': float(final_acc)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics"""
    uptime = (datetime.now() - metrics['start_time']).total_seconds()
    success_rate = (metrics['successful_predictions'] / 
                    max(metrics['total_requests'], 1)) * 100
    
    return jsonify({
        'uptime_seconds': uptime,
        'total_requests': metrics['total_requests'],
        'successful_predictions': metrics['successful_predictions'],
        'failed_predictions': metrics['failed_predictions'],
        'success_rate': success_rate
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)