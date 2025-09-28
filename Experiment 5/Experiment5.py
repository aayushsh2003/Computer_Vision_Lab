# EXPERIMENT 5 - IMAGE SEGMENTATION
# Save all results in "output_segmentation" folder

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
output_folder = "output_segmentation"
os.makedirs(output_folder, exist_ok=True)

# -------------------------
# Load Image
# -------------------------
img_color = cv2.imread('image.jpg')
img_gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# =========================================================
# 1. Threshold-Based Segmentation (Binary Segmentation)
# =========================================================
_, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(output_folder, "binary_segmentation.jpg"), thresh)

plt.imshow(thresh, cmap='gray')
plt.title('Binary Segmentation')
plt.axis('off')
plt.show()

# =========================================================
# 2. Otsu’s Thresholding (Automatic Threshold Selection)
# =========================================================
_, otsu_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(output_folder, "otsu_segmentation.jpg"), otsu_thresh)

plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu's Segmentation")
plt.axis('off')
plt.show()

# =========================================================
# 3. K-means Clustering for Color-Based Segmentation
# =========================================================
Z = img_color.reshape((-1, 3))  # Flatten image
Z = np.float32(Z)

# Define criteria and apply KMeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3  # number of clusters
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((img_color.shape))

cv2.imwrite(os.path.join(output_folder, "kmeans_segmentation.jpg"), segmented_image)

plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('K-Means Segmentation')
plt.axis('off')
plt.show()

# =========================================================
# 4. Watershed Algorithm (Marker-Based Segmentation)
# =========================================================
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Foreground
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(img_color, markers)
img_color[markers == -1] = [255, 0, 0]  # Mark boundaries in red

cv2.imwrite(os.path.join(output_folder, "watershed_segmentation.jpg"), img_color)

plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Watershed Segmentation')
plt.axis('off')
plt.show()

print(f"✅ All segmentation results saved in: {output_folder}")
