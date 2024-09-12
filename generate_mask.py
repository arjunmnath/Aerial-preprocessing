import rasterio
from rasterio.features import rasterize
import fiona
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd


gdb_path = "chat.gdb"
layers = fiona.listlayers(gdb_path)
print(layers)


gdf_build_up = gpd.read_file(gdb_path, layer=layers[0])
gdf_water = gpd.read_file(gdb_path, layer=layers[1])
gdf_road = gpd.read_file(gdb_path, layer=layers[-3])
gdf_bridge = gpd.read_file(gdb_path, layer=layers[4])


def plot_histogram(array: np.array):
    unique_values, counts = np.unique(array, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, counts)
    plt.title('Histogram of Mask')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xticks([0, 10], ['0', '10'])
    plt.show()

def generate_mask(src, geometry):
    size = (src.height, src.width)
    transform = src.transform
    mask = rasterize(
        [(geom, 1) for geom in geometry],
        out_shape=size,
        transform=transform,
        fill=0,  # Background value
        dtype='uint8'
    )
    mask_uint8 = np.where(mask, 255, 0).astype(np.uint8)
    return mask_uint8

with rasterio.open('tifs/meta-preserved-compression-amora-4096x4096.tif') as src:
    binary_class_0 = np.zeros((src.height, src.width))
    binary_class_1 = generate_mask(src, gdf_build_up[gdf_build_up.Roof_type == 1].geometry)
    binary_class_2 = generate_mask(src, gdf_build_up[gdf_build_up.Roof_type == 2].geometry)
    binary_class_3 = generate_mask(src, gdf_build_up[gdf_build_up.Roof_type == 3].geometry)
    binary_class_4 = generate_mask(src, gdf_build_up[gdf_build_up.Roof_type == 4].geometry)
    binary_class_5 = generate_mask(src, gdf_water.geometry)
    binary_class_6 = generate_mask(src, gdf_road.geometry)
    binary_class_7 = generate_mask(src, gdf_bridge.geometry)



# merging all layers
class_images = np.stack([binary_class_0, binary_class_1, binary_class_2, binary_class_3,
                         binary_class_4, binary_class_5 ,binary_class_6,  binary_class_7], axis=-1)


colors = {
    0: (0, 0, 0), # __background__
    1:  (116,238,21), # buildup area type 1
    2: (142,165,255), # buildup area type 2
    3: (240,0,255), # buildup area type 3
    4: (0,30,255), # buildup area type 4
    5: (77,238,234), # water bodies
    6: (255, 231, 0), # roads
    7: (255, 170, 0), # bridges

}


class_indices = np.argmax(class_images, axis= -1) # Class index starts from 1
plot_histogram(class_indices)

# coloring different classes
colored_image = np.zeros((class_indices.shape[0], class_indices.shape[1], 3), dtype=np.uint8)
for class_idx, color in colors.items():
    colored_image[class_indices == class_idx] = color

result_image = Image.fromarray(colored_image)
result_image.show()  # Display the image