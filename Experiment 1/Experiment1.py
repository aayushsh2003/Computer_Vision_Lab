import cv2
import os

# Create a folder to save outputs
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Read image
img = cv2.imread("image.jpg")

# Show & Save Original
cv2.imshow("Original Image", img)
cv2.imwrite(os.path.join(output_folder, "original.jpg"), img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.imwrite(os.path.join(output_folder, "grayscale.jpg"), gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to Binary
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image", binary)
cv2.imwrite(os.path.join(output_folder, "binary.jpg"), binary)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Complement Image
complement = cv2.bitwise_not(img)
cv2.imshow("Complement Image", complement)
cv2.imwrite(os.path.join(output_folder, "complement.jpg"), complement)

cv2.waitKey(0)
cv2.destroyAllWindows()
