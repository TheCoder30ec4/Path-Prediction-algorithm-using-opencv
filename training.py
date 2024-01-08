import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import morphology
from skimage.color import rgb2gray
import scipy.ndimage as nd

# Load the depth frame
depth_frame = np.load('C:/Users/Seshu Reddy/Desktop/1_Depth/Depth_Depth_1677927503980.17236328125000.npy')

# Convert to grayscale
depth_frame_gray = rgb2gray(depth_frame)

# Apply edge segmentation using Canny edge detection
edges = canny(depth_frame_gray)

# Fill regions to perform edge segmentation
filled = nd.binary_fill_holes(edges)

# Calculate the elevation map using the Sobel operator
elevation_map = nd.sobel(depth_frame_gray)

# Create markers for the watershed algorithm by thresholding the elevation map
markers = np.zeros_like(depth_frame_gray)
markers[depth_frame_gray < 0.1171875] = 1 # Lower threshold
markers[depth_frame_gray > 0.5859375] = 2 # Upper threshold

# Perform watershed segmentation
segmentation = morphology.watershed(elevation_map, markers)

# Fill the regions to create a binary mask
mask = nd.binary_fill_holes(segmentation - 1)

# Label the connected components
labelled, _ = nd.label(mask)

# Display the results
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0, 0].imshow(depth_frame)
ax[0, 0].set_title('Original')

ax[0, 1].imshow(edges)
ax[0, 1].set_title('Canny edges')

ax[1, 0].imshow(segmentation)
ax[1, 0].set_title('Segmentation')

ax[1, 1].imshow(labelled)
ax[1, 1].set_title('Labelled')
plt.show()
