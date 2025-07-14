import os
import OpenEXR
import numpy as np
import numpy.typing as npt
from enum import IntEnum


class MoveDirection(IntEnum):
    Backward = -1
    Forward = 1


def read_exr(filepath: str, rotate: bool = False) -> npt.NDArray[np.float16]:
    if not os.path.isfile(filepath):
        raise RuntimeError(f'Incorrect file path: {filepath}')
    
    with OpenEXR.File(filepath) as exr_file:
        header = exr_file.header()
        channels = exr_file.channels()
        min, max = header["dataWindow"]
        # height = max[1] - min[1] + 1
        # width = max[0] - min[1] + 1
        
        channels_data = []
        for channel, values in channels.items():
            pixels = values.pixels

            if rotate:
                new_shape = (1, 0, *(range(2, len(values.pixels.shape))))
                pixels = pixels.transpose(new_shape)
            
            channels_data.append(pixels)

        return np.stack(channels_data, axis=-1) if len(channels_data) > 1 else channels_data[0]
    
def safe_exr(img: npt.NDArray[np.float16], file_path: str) -> None:
    header = {
    "compression" : OpenEXR.ZIP_COMPRESSION,
    "type" : OpenEXR.scanlineimage
    }
    channels = {
        "RGB": img.astype('float16')
    }

    with OpenEXR.File(header, channels) as output:
        output.write(file_path)

def print_pixels(filepath: str, points: list) -> None:
    img = read_exr(filepath)

    print(f'points for {filepath}')
    for point in points:
        print(f'{point=}: {img[point[0], point[1]]}')

def warp_image(image: npt.NDArray[np.float16], flow: npt.NDArray[np.float16], move_dir: MoveDirection) -> npt.NDArray[np.float16]:
    height, width = image.shape[:2]

    warped = np.zeros_like(image)
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    new_y = (y_coords + flow[..., 1] * height * move_dir).astype(np.int32)
    new_x = (x_coords + flow[..., 0] * width * move_dir).astype(np.int32)

    border_mask = (0 <= new_y) & (new_y < height) & (0 <= new_x) & (new_x < width)

    flat_src_y = y_coords[border_mask]
    flat_src_x = x_coords[border_mask]

    flat_dst_y = new_y[border_mask]
    flat_dst_x = new_x[border_mask]

    warped[flat_dst_y, flat_dst_x] = image[flat_src_y, flat_src_x]
    
    return warped
