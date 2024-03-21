import numpy as np
import cv2

# Load data from .npy files
file1_RGB = np.load('20240320_085350.288594_color.npy')
file1_DEPTH = np.load('20240320_085350.288594_depth.npy')
file2_RGB = np.load('20240320_085415.046227_color.npy')
file2_DEPTH = np.load('20240320_085415.046227_depth.npy')
# file1_RGB = np.load('20240320_093036.206357_color.npy')
# file1_DEPTH = np.load('20240320_093036.206357_depth.npy')
# file2_RGB = np.load('20240320_093104.161635_color.npy')
# file2_DEPTH = np.load('20240320_093104.161635_depth.npy')

# Convert depth data to an appropriate format (CV_8U)
file1_DEPTH_uint8 = (file1_DEPTH / np.max(file1_DEPTH) * 255).astype(np.uint8)
file2_DEPTH_uint8 = (file2_DEPTH / np.max(file2_DEPTH) * 255).astype(np.uint8)

# Fill small holes in the depth image using neighborhood color
file1_DEPTH_filled = cv2.inpaint(file1_DEPTH_uint8, (file1_DEPTH_uint8 == 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)
file2_DEPTH_filled = cv2.inpaint(file2_DEPTH_uint8, (file2_DEPTH_uint8 == 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)

# Automatic thresholding
depth_threshold = (np.mean(file1_DEPTH_filled) + np.mean(file2_DEPTH_filled)) / 2 - 50  # Adjusted threshold 20

# Create a mask based on depth information
mask = file2_DEPTH_filled < depth_threshold 
mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

# Replace background parts of the image
fused_image = file1_RGB.copy()
fused_image[mask] = file2_RGB[mask]

# Convert depth data to grayscale images
file1_depth_image_gray_original = (file1_DEPTH / np.max(file1_DEPTH) * 255).astype(np.uint8)
file2_depth_image_gray_original = (file2_DEPTH / np.max(file2_DEPTH) * 255).astype(np.uint8)
file1_depth_image_gray = (file1_DEPTH_filled / np.max(file1_DEPTH_filled) * 255).astype(np.uint8)
file2_depth_image_gray = (file2_DEPTH_filled / np.max(file2_DEPTH_filled) * 255).astype(np.uint8)

# Apply false color to depth images
file1_depth_colored_original = cv2.applyColorMap(file1_depth_image_gray_original, cv2.COLORMAP_JET)
file2_depth_colored_original = cv2.applyColorMap(file2_depth_image_gray_original, cv2.COLORMAP_JET)
file1_depth_colored = cv2.applyColorMap(file1_depth_image_gray, cv2.COLORMAP_JET)
file2_depth_colored = cv2.applyColorMap(file2_depth_image_gray, cv2.COLORMAP_JET)

# Concatenate all images horizontally
depth1 = np.hstack((file1_depth_colored_original, file1_depth_colored))
depth2 = np.hstack((file2_depth_colored_original, file2_depth_colored))
color = np.hstack((file1_RGB, file2_RGB))

# Display the combined image
cv2.imshow('Depth1', depth1)
cv2.imshow('Depth2', depth2)
cv2.imshow('Color', color)
cv2.imshow('Fused', fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
