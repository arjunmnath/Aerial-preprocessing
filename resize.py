import rasterio
from rasterio.enums import Resampling
from affine import Affine
import numpy as np

def resize_tiff_with_aspect_ratio(input_path, output_path, max_size=(1024, 1024)):
    print(f"opening {input_path}...")
    with rasterio.open(input_path) as src:
        original_width, original_height = src.width, src.height
        print("loading transforms...")
        original_transform = src.transform
        print('calculating scales...')
        aspect_ratio = original_width / original_height
        if original_width > original_height:
            new_width = min(max_size[0], original_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_size[1], original_height)
            new_width = int(new_height * aspect_ratio)
        if new_width > max_size[0]:
            new_width = max_size[0]
            new_height = int(new_width / aspect_ratio)
        if new_height > max_size[1]:
            new_height = max_size[1]
            new_width = int(new_height * aspect_ratio)

        scale_x = original_width / new_width
        scale_y = original_height / new_height
        print("computing new transforms...")
        new_transform = original_transform * Affine.scale(scale_x, scale_y)
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.cubic
        )
        metadata = src.meta.copy()
        print("updating metadata...")
        metadata.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })
        print("writing to %s" % output_path)
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(data)


def resize_tiff_with_padding(input_path, output_path, fixed_size=(1024, 1024), fill_value=0):
    print(f"Opening {input_path}...")
    with rasterio.open(input_path) as src:
        original_width, original_height = src.width, src.height
        print("Loading transforms...")
        original_transform = src.transform

        # Calculate the aspect ratio of the original image
        aspect_ratio = original_width / original_height

        # Determine the new size while maintaining aspect ratio
        target_width, target_height = fixed_size
        if aspect_ratio > 1:
            # Landscape orientation: width is the limiting factor
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Portrait orientation: height is the limiting factor
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        print(
            f"Resizing to {new_width}x{new_height} (aspect-ratio maintained) within a {target_width}x{target_height} canvas.")

        # Compute scaling factors
        scale_x = original_width / new_width
        scale_y = original_height / new_height

        # Compute new transform for the resized image
        new_transform = original_transform * Affine.scale(scale_x, scale_y)

        # Resample the image data to the new size
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.cubic
        )

        # Create a blank canvas of the fixed size and fill it with the fill_value (e.g., 0 for black)
        canvas = np.full((src.count, target_height, target_width), fill_value, dtype=data.dtype)

        # Calculate the offsets to center the resized image on the canvas
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        # Place the resized image on the center of the blank canvas
        canvas[:, y_offset:y_offset + new_height, x_offset:x_offset + new_width] = data

        # Update the metadata with new dimensions and transform
        metadata = src.meta.copy()
        print("Updating metadata...")
        metadata.update({
            'height': target_height,
            'width': target_width,
            'transform': original_transform  # Keep the original transform
        })

        # Write the resized and padded image to the output file
        print(f"Writing to {output_path}")
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(canvas)


input_tiff = "tifs/amora.tif"
output_tiff = "tifs/fixed-sized-amora.tif"
# resize_tiff_with_padding(input_tiff, output_tiff, fixed_size=(8192, 8192))
resize_tiff_with_aspect_ratio(input_tiff, output_tiff, max_size=(8192, 8192))
