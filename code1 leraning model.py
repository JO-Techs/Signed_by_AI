import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def preprocess_signature(image_path):
    """
    Preprocess the signature image: grayscale, thresholding, and keypoint detection.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Adaptive Thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Extract Keypoints and Descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(binary, None)
    
    return keypoints, descriptors

def store_reference_signature(image_path, storage_path="reference_signature.npy"):
    """
    Process and store the reference signature's descriptors for future matching.
    """
    _, descriptors = preprocess_signature(image_path)
    if descriptors is not None:
        np.save(storage_path, descriptors)
        print("Reference signature stored successfully.")
    else:
        print("No features detected. Ensure the image contains a clear signature.")

def match_signature(input_image_path, reference_path="reference_signature.npy"):
    """
    Match an input signature with the stored reference signature.
    """
    # Load the stored reference signature descriptors
    if not os.path.exists(reference_path):
        print("Reference signature not found! Please store one first.")
        return
    
    reference_descriptors = np.load(reference_path)
    _, input_descriptors = preprocess_signature(input_image_path)
    
    if input_descriptors is None:
        print("No features detected in input signature.")
        return

    # Compare descriptors using cosine similarity
    similarity_scores = []
    for ref_desc in reference_descriptors:
        for input_desc in input_descriptors:
            similarity = cosine_similarity(ref_desc.reshape(1, -1), input_desc.reshape(1, -1))
            similarity_scores.append(similarity[0][0])
    
    # Average similarity score
    avg_similarity = np.mean(similarity_scores)
    print(f"Similarity Score: {avg_similarity:.2f}")

    # Define a threshold for matching
    threshold = 0.7  # You can fine-tune this value
    if avg_similarity > threshold:
        print("Signature is authentic.")
    else:
        print("Signature does not match.")

if __name__ == "__main__":
    print("1. Store a reference signature")
    print("2. Match a new signature against the reference")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        image_path = input("Enter the path to the reference signature image: ")
        store_reference_signature(image_path)
    elif choice == "2":
        input_path = input("Enter the path to the signature to verify: ")
        match_signature(input_path)
    else:
        print("Invalid choice.")
