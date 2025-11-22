"""Model creation and training"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pickle
from datetime import datetime


def create_model(input_shape=(32, 32, 3), num_classes=10):
    """Create CNN model with transfer learning"""
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
        layers.Dense(256, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=50, batch_size=128, model_path='best_model.h5'):
    """Train model with callbacks"""
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, 
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1)
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
                  epochs=10, batch_size=32):
    """Retrain existing model with new data"""
    model = keras.models.load_model(model_path)
    
    print(f"Loaded model from {model_path}")
    print(f"Retraining with {len(new_X_train)} new samples...")
    
    # Unfreeze last layers for fine-tuning
    for layer in model.layers[-10:]:
        layer.trainable = True
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        new_X_train, new_y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_model_path = f'models/retrained_model_{timestamp}.h5'
    model.save(new_model_path)
    
    return model, history, new_model_path