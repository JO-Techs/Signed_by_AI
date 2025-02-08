import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, threshold1=100, threshold2=200)

plt.figure(figsize=(8, 6))
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.title("Canny Edge Detection")
plt.show()
