"""
Export CIFAR-10 test images for demo
Creates sample images that work well with the model
"""
from tensorflow import keras
import matplotlib.pyplot as plt
import os

print("=" * 70)
print("EXPORTING CIFAR-10 TEST IMAGES")
print("=" * 70)

# Load test data
print("\n Loading CIFAR-10 test data...")
(_, _), (X_test, y_test) = keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Loaded {len(X_test)} test images")

# Create directory
output_dir = 'cifar10_test_images'
os.makedirs(output_dir, exist_ok=True)
print(f"\n Created directory: {output_dir}/")

# Export 2 images per class
print("\nExporting test images...")
print("-" * 70)

for class_idx, class_name in enumerate(class_names):
    # Find images of this class
    class_indices = [i for i, label in enumerate(y_test) if label[0] == class_idx]
    
    # Save first 2
    for i in range(2):
        img_idx = class_indices[i]
        img = X_test[img_idx]
        
        filename = f'{output_dir}/{class_name}_{i+1}.png'
        plt.imsave(filename, img)
        print(f"✅ Saved: {filename}")

print("\n" + "=" * 70)
print("EXPORT COMPLETE!")
print("=" * 70)
print(f"\n Location: {output_dir}/")
print(f"Total images: 20 (2 per class)")
print("\n Use these images in your demo to show correct predictions!")
print("=" * 70)