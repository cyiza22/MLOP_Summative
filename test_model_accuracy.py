"""
Test Model Accuracy on CIFAR-10
Validates that the model achieves stated accuracy on test data
"""
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

print("=" * 70)
print("CIFAR-10 MODEL VALIDATION")
print("=" * 70)

# Load model
print("\nLoading model...")
model = keras.models.load_model('models/cifar10_classifier.h5')
print("Model loaded successfully")

# Load CIFAR-10 test data
print("\n Loading CIFAR-10 test dataset...")
(_, _), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_test = X_test.astype('float32') / 255.0
y_test = y_test.flatten()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Loaded {len(X_test)} test images")

# Make predictions
print("\n🔮 Making predictions on test set...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("=" * 70)

# Per-class accuracy
print("\n Per-Class Performance:")
print("-" * 70)
for i, class_name in enumerate(class_names):
    class_mask = y_test == i
class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
print(f"{class_name:12s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

# Detailed report
print("\n Detailed Classification Report:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=class_names))

# Validation check
print("\n" + "=" * 70)
if accuracy >= 0.85:
    print("VALIDATION PASSED: Model achieves expected performance on CIFAR-10")
else:
    print(" WARNING: Model accuracy below expected threshold")
print("=" * 70)
