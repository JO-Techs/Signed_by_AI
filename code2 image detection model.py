import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_signature_features(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found. Please provide a valid image path.")
    
    # Resize the image for better processing (optional)
    image = cv2.resize(image, (800, 400))
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Use adaptive thresholding for better edge detection
    binary_image = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(binary_image, 50, 150)
    
    # Detect contours from the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize feature storage
    feature_descriptions = []
    
    for contour in contours:
        # Calculate properties like area, perimeter, and curvature
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx_curve = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Store features for each contour
        feature_descriptions.append({
            "area": area,
            "perimeter": perimeter,
            "curves": len(approx_curve)
        })
    
    # Display the processed images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Binary Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.show()
    
    return feature_descriptions

# Provide the path to the image containing the signature
image_path = "signature_sample.jpg"  # Replace with your signature image path
features = extract_signature_features(image_path)

# Print extracted features
for i, feature in enumerate(features):
    print(f"Contour {i+1}: Area = {feature['area']:.2f}, Perimeter = {feature['perimeter']:.2f}, Curves = {feature['curves']}")
