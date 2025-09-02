import cv2
import matplotlib.pyplot as plt
import os
from skimage.feature import hog
from skimage import color, exposure

# ---------------- Create Output Folder ----------------
output_folder = "feature_output"
os.makedirs(output_folder, exist_ok=True)

# ---------------- 1. SIFT Feature Extraction ----------------
# Load image in grayscale
img_gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Initialize SIFT
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(img_gray, None)

# Draw keypoints
img_sift = cv2.drawKeypoints(img_gray, keypoints, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save SIFT result
cv2.imwrite(os.path.join(output_folder, "sift_keypoints.jpg"), img_sift)

# Also save with matplotlib (optional, better color handling)
plt.imshow(img_sift, cmap="gray")
plt.title("SIFT Keypoints")
plt.axis("off")
plt.savefig(os.path.join(output_folder, "sift_keypoints_plot.png"))
plt.close()

# ---------------- 2. HOG Feature Extraction ----------------
# Load image in BGR
img = cv2.imread("image.jpg")

# Convert to grayscale using skimage
gray = color.rgb2gray(img)

# Extract HOG features + visualization
features, hog_image = hog(gray, pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2),
                          visualize=True, channel_axis=None)

# Rescale for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Save grayscale image
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")
plt.savefig(os.path.join(output_folder, "hog_original_gray.png"))
plt.close()

# Save HOG visualization
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")
plt.savefig(os.path.join(output_folder, "hog_visualization.png"))
plt.close()

# Save side-by-side comparison
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1), plt.imshow(gray, cmap="gray"), plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2), plt.imshow(hog_image_rescaled, cmap="gray"), plt.title("HOG Visualization")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "hog_comparison.png"))
plt.close()
