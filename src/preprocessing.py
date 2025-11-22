
"""
Data preprocessing utilities for image classification
"""
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2

def preprocess_image(image_path, target_size=(32, 32)):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to image file or numpy array
        target_size: Target size for the image (height, width)
    
    Returns:
        Preprocessed image array normalized to [0, 1]
    """
    if isinstance(image_path, str):
        # Load image from file
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
    elif isinstance(image_path, np.ndarray):
        # Resize if needed
        if image_path.shape[:2] != target_size:
            img_array = cv2.resize(image_path, target_size)
        else:
            img_array = image_path
    else:
        raise ValueError("image_path must be a string path or numpy array")
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def preprocess_batch(image_paths, target_size=(32, 32)):
    """
    Preprocess a batch of images
    
    Args:
        image_paths: List of image paths or numpy arrays
        target_size: Target size for images
    
    Returns:
        Batch of preprocessed images
    """
    images = []
    for img_path in image_paths:
        img = preprocess_image(img_path, target_size)
        images.append(img)
    
    return np.array(images)

def augment_image(image):
    """
    Apply data augmentation to an image
    
    Args:
        image: Image array
    
    Returns:
        Augmented image
    """
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = np.fliplr(image)
    
    # Random rotation
    angle = np.random.randint(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    
    return image


# ============================================
# FILE: src/model.py
# ============================================
"""
Model creation and training utilities
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pickle
from datetime import datetime

def create_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a CNN model using Transfer Learning with MobileNetV2
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=50, batch_size=128, model_path='best_model.h5'):
    """
    Train the model with optimization callbacks
    """
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def retrain_model(model_path, new_X_train, new_y_train, X_val, y_val,
                  epochs=10, batch_size=128):
    """
    Retrain an existing model with new data
    
    Args:
        model_path: Path to existing model
        new_X_train: New training data
        new_y_train: New training labels
        X_val: Validation data
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Retrained model and training history
    """
    # Load existing model
    model = keras.models.load_model(model_path)
    
    print(f"Loaded model from {model_path}")
    print(f"Retraining with {len(new_X_train)} new samples...")
    
    # Unfreeze some layers for fine-tuning
    for layer in model.layers[-10:]:
        layer.trainable = True
    
    # Compile with lower learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    # Train
    history = model.fit(
        new_X_train, new_y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save retrained model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_model_path = f'retrained_model_{timestamp}.h5'
    model.save(new_model_path)
    
    print(f"Retrained model saved to {new_model_path}")
    
    return model, history, new_model_path


# ============================================
# FILE: src/prediction.py
# ============================================
"""
Prediction utilities
"""
import numpy as np
from tensorflow import keras
import pickle

class ModelPredictor:
    """
    Class for making predictions with a trained model
    """
    
    def __init__(self, model_path, metadata_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            metadata_path: Path to model metadata (optional)
        """
        self.model = keras.models.load_model(model_path)
        self.class_names = None
        
        if metadata_path:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.class_names = metadata.get('class_names', None)
        
        if self.class_names is None:
            # Default CIFAR-10 classes
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                               'dog', 'frog', 'horse', 'ship', 'truck']
    
    def predict(self, image):
        """
        Predict the class of a single image
        
        Args:
            image: Image array (32, 32, 3) normalized to [0, 1]
        
        Returns:
            Dictionary with prediction results
        """
        if image.shape != (32, 32, 3):
            raise ValueError(f"Image must be shape (32, 32, 3), got {image.shape}")
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get all class probabilities
        all_predictions = [
            {
                'class': self.class_names[i], 
                'confidence': float(predictions[0][i])
            }
            for i in range(len(self.class_names))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'predicted_class': self.class_names[predicted_class_idx],
            'class_index': int(predicted_class_idx),
            'confidence': float(confidence),
            'all_predictions': all_predictions
        }
    
    def predict_batch(self, images):
        """
        Predict classes for a batch of images
        
        Args:
            images: Batch of images (N, 32, 32, 3)
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        
        return results


# ============================================
# FILE: app.py (Flask API)
# ============================================
"""
Flask API for ML model serving
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime
import tensorflow as tf
from src.preprocessing import preprocess_image, preprocess_batch
from src.prediction import ModelPredictor
from src.model import retrain_model
import pickle

app = Flask(__name__)
CORS(app)

# Initialize predictor
MODEL_PATH = 'models/cifar10_classifier.h5'
METADATA_PATH = 'models/model_metadata.pkl'
predictor = ModelPredictor(MODEL_PATH, METADATA_PATH)

# Create directories
os.makedirs('data/uploads', exist_ok=True)
os.makedirs('data/train', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Metrics tracking
metrics = {
    'total_requests': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'start_time': datetime.now()
}

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    uptime = (datetime.now() - metrics['start_time']).total_seconds()
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': uptime,
        'model_loaded': predictor.model is not None,
        'total_requests': metrics['total_requests']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint for single image
    Expects: image file in request
    """
    metrics['total_requests'] += 1
    
    try:
        # Check if image in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read and preprocess image
        img = Image.open(file.stream).convert('RGB')
        img_array = np.array(img)
        processed_img = preprocess_image(img_array)
        
        # Predict
        result = predictor.predict(processed_img)
        
        metrics['successful_predictions'] += 1
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        metrics['failed_predictions'] += 1
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict endpoint for multiple images
    """
    metrics['total_requests'] += 1
    
    try:
        files = request.files.getlist('images')
        
        if not files:
            return jsonify({'error': 'No images provided'}), 400
        
        # Process all images
        images = []
        for file in files:
            img = Image.open(file.stream).convert('RGB')
            img_array = np.array(img)
            processed_img = preprocess_image(img_array)
            images.append(processed_img)
        
        # Batch predict
        results = predictor.predict_batch(images)
        
        metrics['successful_predictions'] += len(results)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        metrics['failed_predictions'] += 1
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """
    Upload new training data for retraining
    """
    try:
        files = request.files.getlist('images')
        labels = request.form.getlist('labels')
        
        if not files:
            return jsonify({'error': 'No images provided'}), 400
        
        # Save uploaded files
        saved_files = []
        for file, label in zip(files, labels):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{label}_{timestamp}_{file.filename}"
            filepath = os.path.join('data/train', filename)
            file.save(filepath)
            saved_files.append(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Uploaded {len(saved_files)} images',
            'files': saved_files
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Trigger model retraining with uploaded data
    """
    try:
        # Load training data from data/train directory
        train_files = os.listdir('data/train')
        
        if not train_files:
            return jsonify({'error': 'No training data available'}), 400
        
        # Load and preprocess training images
        X_new = []
        y_new = []
        
        for filename in train_files:
            filepath = os.path.join('data/train', filename)
            label_name = filename.split('_')[0]
            
            # Get label index
            if label_name in predictor.class_names:
                label_idx = predictor.class_names.index(label_name)
            else:
                continue
            
            # Load and preprocess image
            img = preprocess_image(filepath)
            X_new.append(img)
            y_new.append(label_idx)
        
        X_new = np.array(X_new)
        y_new = tf.keras.utils.to_categorical(y_new, len(predictor.class_names))
        
        # Use a portion of existing test data as validation
        (_, _), (X_val, y_val) = tf.keras.datasets.cifar10.load_data()
        X_val = X_val[:1000].astype('float32') / 255.0
        y_val = tf.keras.utils.to_categorical(y_val[:1000], 10)
        
        # Retrain model
        new_model, history, new_model_path = retrain_model(
            MODEL_PATH, X_new, y_new, X_val, y_val,
            epochs=10, batch_size=32
        )
        
        # Update predictor with new model
        predictor.model = new_model
        
        # Save new metrics
        final_accuracy = history.history['val_accuracy'][-1]
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'new_model_path': new_model_path,
            'training_samples': len(X_new),
            'final_accuracy': float(final_accuracy),
            'epochs': len(history.history['accuracy'])
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get system metrics
    """
    uptime = (datetime.now() - metrics['start_time']).total_seconds()
    
    return jsonify({
        'uptime_seconds': uptime,
        'total_requests': metrics['total_requests'],
        'successful_predictions': metrics['successful_predictions'],
        'failed_predictions': metrics['failed_predictions'],
        'success_rate': (metrics['successful_predictions'] / max(metrics['total_requests'], 1)) * 100
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# ============================================
# FILE: locustfile.py (Load Testing)
# ============================================
"""
Locust load testing configuration
"""
from locust import HttpUser, task, between
import random

class ModelUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def predict_single_image(self):
        """Test single prediction endpoint"""
        # Generate random image data
        import numpy as np
        from PIL import Image
        import io
        
        # Create random image
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Send request
        files = {'image': ('test.png', img_bytes, 'image/png')}
        self.client.post("/predict", files=files)
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/health")
    
    @task(1)
    def get_metrics(self):
        """Test metrics endpoint"""
        self.client.get("/metrics")