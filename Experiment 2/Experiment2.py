import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Create output folder
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# 1. Read Image
img = cv2.imread("image.jpg")

# ---------------- Contrast Adjustment ----------------
alpha = 1.5  # Contrast control
beta = 0     # Brightness control
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

cv2.imshow("Contrast Adjusted", adjusted)
cv2.imwrite(os.path.join(output_folder, "contrast_adjusted.jpg"), adjusted)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------- Histogram Calculation ----------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure()
plt.plot(hist, color='black')
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_folder, "histogram.png"))
plt.show()

# ---------------- Histogram Equalization ----------------
equalized = cv2.equalizeHist(gray)

cv2.imshow("Original Gray", gray)
cv2.imshow("Equalized Image", equalized)
cv2.imwrite(os.path.join(output_folder, "gray.jpg"), gray)
cv2.imwrite(os.path.join(output_folder, "equalized.jpg"), equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------- Compare Histograms ----------------
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.hist(gray.ravel(), 256, [0, 256], color='gray')
plt.title("Original Histogram")

plt.subplot(1, 2, 2)
plt.hist(equalized.ravel(), 256, [0, 256], color='black')
plt.title("Equalized Histogram")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "histogram_comparison.png"))
plt.show()
