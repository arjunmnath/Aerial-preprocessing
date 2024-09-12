import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin
import fiona
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import geopandas as gpd
from shapely.geometry import mapping

gdb_path = "chat.gdb"
layers = fiona.listlayers(gdb_path)
print(layers)
gdf = gpd.read_file(gdb_path, layer=layers[0])


def generate_outline(src):
        out_shape = (src.height, src.width)
        out_transform = src.transform
        out_meta = src.meta.copy()
        out_meta.update({
            'dtype': 'uint8',  # You can use a simple integer type for binary boundary output
            'count': 1,
            'compress': 'lzw'  # Optional: Compression to save space
        })

        geometries = [mapping(geom) for geom in gdf.boundary]

        rasterized_boundaries = rasterize(
            geometries,
            out_shape=out_shape,
            transform=out_transform,
            fill=0,  # Background value
            default_value=1  # Value for the polygon boundaries
        )
        mask_uint8 = np.where(rasterized_boundaries, 255, 0).astype(np.uint8)
        return mask_uint8

with rasterio.open('tifs/meta-preserved-compression-amora-4096x4096.tif') as src:
    image = generate_outline(src)
result_image = Image.fromarray(image)
print(result_image.size)

result_image.show()  # Display the image
result_image.save("outlines/amora.png")