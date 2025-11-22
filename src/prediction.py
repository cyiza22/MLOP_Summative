"""Prediction utilities"""
import numpy as np
from tensorflow import keras


class ModelPredictor:
    """Class for making predictions"""
    
    def __init__(self, model_path, class_names=None):
        self.model = keras.models.load_model(model_path)
        
        if class_names is None:
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 
                                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            self.class_names = class_names
    
    def predict(self, image):
        """Predict single image"""
        if image.shape != (32, 32, 3):
            raise ValueError(f"Image must be (32,32,3), got {image.shape}")
        
        image_batch = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image_batch, verbose=0)
        
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        all_predictions = [
            {'class': self.class_names[i], 'confidence': float(predictions[0][i])}
            for i in range(len(self.class_names))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'predicted_class': self.class_names[predicted_idx],
            'class_index': int(predicted_idx),
            'confidence': float(confidence),
            'all_predictions': all_predictions
        }
    
    def predict_batch(self, images):
        """Predict batch of images"""
        return [self.predict(img) for img in images]
