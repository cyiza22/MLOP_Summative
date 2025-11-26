"""
Data preprocessing utilities for image classification
Handles automatic resizing and optimization for CIFAR-10 model
"""
import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image_path, target_size=(32, 32)):
    """
    Preprocess a single image for prediction with optimal resizing
    
    This function ensures images are properly prepared for the CIFAR-10 model:
    1. Converts to RGB (handles grayscale/RGBA)
    2. Smart crops to square (preserves aspect ratio)
    3. Resizes to 32x32 using high-quality interpolation
    4. Normalizes pixel values to [0, 1]
    
    Args:
        image_path: Path to image file or numpy array
        target_size: Target size (height, width) - default (32, 32)
    
    Returns:
        Preprocessed image array normalized to [0, 1]
    """
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, np.ndarray):
        # Convert numpy array to PIL Image
        if image_path.dtype == np.float32 or image_path.dtype == np.float64:
            # If already normalized, scale back
            if image_path.max() <= 1.0:
                image_path = (image_path * 255).astype(np.uint8)
        img = Image.fromarray(image_path.astype(np.uint8))
    else:
        raise ValueError("image_path must be a string path or numpy array")
    
    # Get original size
    original_width, original_height = img.size
    
    # STEP 1: Center crop to square first (preserves important features)
    # This prevents distortion that occurs when directly resizing rectangular images
    min_dimension = min(original_width, original_height)
    
    # Calculate crop box (centered)
    left = (original_width - min_dimension) // 2
    top = (original_height - min_dimension) // 2
    right = left + min_dimension
    bottom = top + min_dimension
    
    # Crop to square
    img = img.crop((left, top, right, bottom))
    
    # STEP 2: Resize to target size using high-quality interpolation
    # LANCZOS provides best quality for downsampling
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # STEP 3: Convert to numpy array
    img_array = np.array(img)
    
    # STEP 4: Normalize to [0, 1] range
    img_array = img_array.astype('float32') / 255.0
    
    return img_array


def preprocess_batch(image_paths, target_size=(32, 32)):
    """
    Preprocess a batch of images
    
    Args:
        image_paths: List of image paths or numpy arrays
        target_size: Target size for images
    
    Returns:
        Batch of preprocessed images as numpy array
    """
    images = []
    for img_path in image_paths:
        img = preprocess_image(img_path, target_size)
        images.append(img)
    
    return np.array(images)


def validate_image(image_path):
    """
    Validate that an image can be processed
    
    Args:
        image_path: Path to image file
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if isinstance(image_path, str):
            img = Image.open(image_path)
        elif isinstance(image_path, np.ndarray):
            return True, "Valid numpy array"
        else:
            return False, "Invalid image format"
        
        # Check if image can be converted to RGB
        img.convert('RGB')
        
        # Check minimum size
        width, height = img.size
        if width < 32 or height < 32:
            return False, f"Image too small: {width}x{height}. Minimum size is 32x32."
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Error processing image: {str(e)}"


def get_preprocessing_info(image_path):
    """
    Get information about how an image will be preprocessed
    
    Args:
        image_path: Path to image file
    
    Returns:
        dict: Information about preprocessing steps
    """
    try:
        if isinstance(image_path, str):
            img = Image.open(image_path)
        elif isinstance(image_path, np.ndarray):
            img = Image.fromarray(image_path.astype(np.uint8))
        else:
            return {"error": "Invalid image format"}
        
        original_width, original_height = img.size
        min_dimension = min(original_width, original_height)
        
        # Calculate how much will be cropped
        crop_width = original_width - min_dimension
        crop_height = original_height - min_dimension
        
        # Calculate resize ratio
        resize_ratio = min_dimension / 32.0
        
        return {
            "original_size": f"{original_width}×{original_height}",
            "crop_amount": f"{crop_width}px width, {crop_height}px height",
            "cropped_size": f"{min_dimension}×{min_dimension}",
            "final_size": "32×32",
            "resize_ratio": f"{resize_ratio:.1f}x reduction",
            "pixels_original": original_width * original_height,
            "pixels_final": 32 * 32,
            "data_retention": f"{(1024 / (original_width * original_height) * 100):.2f}%"
        }
        
    except Exception as e:
        return {"error": str(e)}