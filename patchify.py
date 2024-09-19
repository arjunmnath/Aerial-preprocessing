import numpy as np
from PIL import Image
import cv2
from numpy import dtype
import matplotlib.pyplot as plt


def create_tiles(image: np.ndarray,tile_size=(512, 512), stride=(256, 256), channels=3):
    # when tile_size == stride, overlapping = 0
    height, width, _ = image.shape
    tile_id = 0
    num_patches_height = (height - tile_size[1]) // stride[1] + 1
    num_patches_width = (width - tile_size[0]) // stride[0] + 1
    total_patches = num_patches_height * num_patches_width
    _patches = np.empty((total_patches, *tile_size, channels), dtype=np.uint8)
    for y in range(0, height - tile_size[1] + 1, stride[1]):
        for x in range(0, width - tile_size[0] + 1, stride[0]):
            _patches[tile_id] = image[y:y + tile_size[1], x:x + tile_size[0]]
            tile_id += 1
    return _patches


def merge_tiles(patches: np.ndarray, tile_size=(512, 512), stride=(256, 256), channels=3):
    # Extract number of patches, tile height, tile width, and channels from the patches array
    num_patches, tile_height, tile_width, channels = patches.shape

    # Calculate the number of patches along height and width
    num_patches_height = (patches.shape[0] * stride[1] - tile_size[1]) // stride[1] + 1
    num_patches_width = num_patches // num_patches_height

    # Calculate original image dimensions
    height = tile_height + (num_patches_height - 1) * stride[1]
    width = tile_width + (num_patches_width - 1) * stride[0]

    # Initialize an empty array for the merged image
    merged_image = np.zeros((height, width, channels), dtype=np.uint8)

    # Reconstruct the image from patches
    tile_id = 0
    for y in range(0, height - tile_height + 1, stride[1]):
        for x in range(0, width - tile_width + 1, stride[0]):
            merged_image[y:y + tile_height, x:x + tile_width] = patches[tile_id]
            tile_id += 1

    return merged_image

# Example usage
# image_path = 'tifs/meta-preserved-compression-amora-4096x4096.tif'
image_path = 'tifs/fixed-sized-amora.tif'
mask_path = "amora-mask.tif"
image = cv2.imread(image_path)
mask = cv2.imread(mask_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
size = (1024, 1024)
stride = (512, 512)
patches = create_tiles(image)
mask_patches = create_tiles(mask)
i =  239
ariel_img = Image.fromarray(patches[i])
mask_img = Image.fromarray(mask_patches[i])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# First image
axes[0].imshow(ariel_img)
axes[0].set_title('Input Patch (512x512)')  # Add label/title
# axes[0].axis('off')  # Turn off axis

# Second image
axes[1].imshow(mask_img)
axes[1].set_title('Predicted Patch (512x512)')  # Add label/title
# axes[1].axis('off')  # Turn off axis

# Adjust layout for a cleaner look
plt.savefig('side_by_side_images.png', format='png')

plt.show()
