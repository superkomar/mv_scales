import os
import OpenEXR
import numpy as np
import numpy.typing as npt
from enum import IntEnum
from typing import Tuple

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


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
    
def write_exr(img: npt.NDArray[np.float32], file_path: str) -> None:
    header = {
        "compression" : OpenEXR.ZIP_COMPRESSION,
        "type" : OpenEXR.scanlineimage
    }

    if img.shape[2] == 2:
        channels = {
            'R': img[..., 0].astype('float16'),
            'G': img[..., 1].astype('float16')
        }
    else:
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


class ImageUtils:
    @staticmethod
    def normalize(image: npt.NDArray[np.float16]) -> npt.NDArray[np.float16]:
        min = image.min()
        max = image.max()

        normalized = (image - min) / (max - min)
        return normalized
    
    @staticmethod
    def make_8bit(image: npt.NDArray[np.float16]) -> npt.NDArray[np.uint8]:
        min = image.min()
        max = image.max()

        norm_image = (image - min) / (max - min)
        norm_image = (norm_image * 255.0)

        image_8bit = np.sum(norm_image, axis=-1) / image.shape[-1]
        
        return image_8bit.astype(np.uint8)
    
    @staticmethod
    def make_grayscale(img: npt.NDArray[np.float16]) -> npt.NDArray[np.uint8]:
        tone_mapping = cv2.createTonemapDrago(gamma=2.5, bias=0.85)
        tone_mapped_img = tone_mapping.process((img[:,:,:3]).astype(np.float32))
        tone_mapped_img = ImageUtils.remove_nans(tone_mapped_img)
        tone_mapped_img = (tone_mapped_img * 255).astype(np.uint8)

        grayscale = cv2.cvtColor(tone_mapped_img, cv2.COLOR_RGB2GRAY)

        return grayscale
    
    @staticmethod
    def linear_interpolation(coord: float) -> int:
        coord_0 = int(coord)
        coord_1 = coord_0 + 1
        
        d_coord = coord - coord_0

        if d_coord < 0.5:
            result = coord_0
        else:
            result = coord_1

        return result
    
    @staticmethod
    def bilinear_interpolation(x: float, y: float) -> Tuple[int, int]:
        dx = x - int(x)
        dy = y - int(y)

        x_0 = int(x)
        y_0 = int(y)
        x_1 = x_0 + 1
        y_1 = y_0 + 1

        w_00 = (1 - dx) * (1 - dy)
        w_01 = (1 - dx) * dy
        w_10 = dx * (1 - dy)
        w_11 = dx * dy

        quads = [
            ((x_0, y_0), w_00),
            ((x_0, y_1), w_01),
            ((x_1, y_0), w_10),
            ((x_1, y_1), w_11)
        ]

        return max(quads, key=lambda x: x[1])[0]

    @staticmethod
    def remove_nans(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        nan_checks = np.isnan(image)
        nan_pixels = np.argwhere(nan_checks)

        for pixel in nan_pixels:
            image[pixel[0], pixel[1], pixel[2]] = 0.0

        return image
    
    @staticmethod
    def warp_image(
            image: npt.NDArray[np.float16], flow: npt.NDArray[np.float16], is_moving_forward: bool = True
        ) -> npt.NDArray[np.float16]:

        height, width = image.shape[:2]

        warped = np.zeros_like(image)
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        move_dir = 1 if is_moving_forward else -1

        new_y = (y_coords + flow[..., 1] * height * move_dir).astype(np.int32)
        new_x = (x_coords + flow[..., 0] *  width * move_dir).astype(np.int32)

        border_mask = (0 <= new_y) & (new_y < height) & (0 <= new_x) & (new_x < width)

        flat_src_y = y_coords[border_mask]
        flat_src_x = x_coords[border_mask]

        flat_dst_y = new_y[border_mask]
        flat_dst_x = new_x[border_mask]

        warped[flat_dst_y, flat_dst_x] = image[flat_src_y, flat_src_x]
        
        return warped

    @staticmethod
    def diff_images(
        img_1: npt.NDArray[np.float32], img_2: npt.NDArray[np.float32], diff_eps: float = 1.0e-6
    ) -> npt.NDArray[np.float32]:
        norm_img_1 = ImageUtils.normalize(img_1)
        norm_img_2 = ImageUtils.normalize(img_2)

        diff = np.abs(norm_img_1 - norm_img_2)
        diff_mask = diff > diff_eps
        
        zero_diff = diff[diff_mask]

        diff[zero_diff] = 0.0

        return diff

    @staticmethod
    def diff_images_as_grayscale(img_1: npt.NDArray[np.float32], img_2: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        gray_1 = ImageUtils.make_8bit(img_1)
        gray_2 = ImageUtils.make_8bit(img_2)

        return np.abs(gray_1 - gray_2)


