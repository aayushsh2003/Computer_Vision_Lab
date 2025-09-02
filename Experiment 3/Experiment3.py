import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
output_folder = "fft_output"
os.makedirs(output_folder, exist_ok=True)

# ---------------- Step 1: Read image and convert to grayscale ----------------
img = cv2.imread("image.jpg", 0)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Save original
cv2.imwrite(os.path.join(output_folder, "original.jpg"), img)

# ---------------- Step 2: Perform 2D Fourier Transform ----------------
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)  # Shift zero freq to center
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # add +1 to avoid log(0)

plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("FFT Spectrum")
plt.axis("off")
plt.savefig(os.path.join(output_folder, "fft_spectrum.png"))
plt.close()

# ---------------- Step 3: Low-Pass Filter ----------------
mask_low = np.zeros((rows, cols), np.uint8)
r = 30  # Radius for low-pass filter
mask_low[crow-r:crow+r, ccol-r:ccol+r] = 1

fshift_low = fshift * mask_low
f_ishift_low = np.fft.ifftshift(fshift_low)
img_low = np.fft.ifft2(f_ishift_low)
img_low = np.abs(img_low)

cv2.imwrite(os.path.join(output_folder, "low_pass_filtered.jpg"), img_low.astype(np.uint8))

# ---------------- Step 4: High-Pass Filter ----------------
mask_high = np.ones((rows, cols), np.uint8)
mask_high[crow-r:crow+r, ccol-r:ccol+r] = 0  # Block low freq

fshift_high = fshift * mask_high
f_ishift_high = np.fft.ifftshift(fshift_high)
img_high = np.fft.ifft2(f_ishift_high)
img_high = np.abs(img_high)

cv2.imwrite(os.path.join(output_folder, "high_pass_filtered.jpg"), img_high.astype(np.uint8))

# ---------------- Step 5: Combined Plot for Reference ----------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("FFT Spectrum")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_low, cmap="gray")
plt.title("Low-Pass Filtered")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "comparison_lowpass.png"))
plt.close()

# Another figure for high-pass comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("FFT Spectrum")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_high, cmap="gray")
plt.title("High-Pass Filtered")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "comparison_highpass.png"))
plt.close()
