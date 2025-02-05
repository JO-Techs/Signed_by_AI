"""
Laplacian Sharpening Algorithm

This script applies Laplacian sharpening to an image to enhance its edges. 
It works by computing the Laplacian of the image, scaling it, and subtracting it from the original image.

Steps:
1. Load the image in grayscale.
2. Apply the Laplacian filter to detect edges.
3. Subtract the Laplacian from the original image to enhance edges.
4. Normalize and display the sharpened image.

Dependencies:
- OpenCV
- NumPy

Usage:
- Place an image in the same directory as the script.
- Run the script to visualize the sharpened output.
"""

import cv2
import numpy as np

def laplacian_sharpening(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found")
    
    laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    sharpened = cv2.subtract(img, 0.5 * laplacian)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    cv2.imshow("Original Image", img)
    cv2.imshow("Laplacian Edge Map", laplacian)
    cv2.imshow("Sharpened Image", sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

laplacian_sharpening("image.jpg")
