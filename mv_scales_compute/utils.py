import os
import OpenEXR
import numpy as np
import numpy.typing as npt
from typing import Tuple

import torch
import torch.nn.functional as F

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


class ExrUtils:
    @staticmethod
    def read_image(filepath: str) -> npt.NDArray[np.float32]:
        return ExrUtils._read_exr(filepath)

    @staticmethod
    def read_motion_vectors(filepath: str, rotate_xy: bool = False) -> npt.NDArray[np.float32]:
        motion_vectors = ExrUtils._read_exr(filepath)

        motion_vectors = motion_vectors[..., :2]

        if rotate_xy:
            motion_vectors = motion_vectors[..., ::-1]

        return motion_vectors
    
    @staticmethod
    def _read_exr(filepath: str) -> npt.NDArray[np.float32]:
        if not os.path.isfile(filepath):
            raise RuntimeError(f'Incorrect file path: {filepath}')
        
        with OpenEXR.File(filepath) as exr_file:
            channels_exr = exr_file.channels()
            channels = [np.array(value.pixels, dtype=np.float32) for value in channels_exr.values()]

            return np.stack(channels, axis=-1) if len(channels) > 1 else channels[0]
    
    @staticmethod
    def write_exr(img: npt.NDArray[np.float32], file_path: str) -> None:
        header = {
            "compression" : OpenEXR.ZIP_COMPRESSION,
            "type" : OpenEXR.scanlineimage
        }

        if len(img.shape) == 2 or img.shape[2] == 1:
            channels = {
                "RGB": img.astype('float16')
            }

        elif img.shape[2] == 2:
            channels = {
                'R': img[..., 0].astype('float16'),
                'G': img[..., 1].astype('float16'),
            }
        else:
            channels = {
                'R': img[..., 0].astype('float16'),
                'G': img[..., 1].astype('float16'),
                'B': img[..., 2].astype('float16'),
            }

        with OpenEXR.File(header, channels) as output:
            output.write(file_path)

    @staticmethod
    def get_pixels_value(image: npt.NDArray[np.float32], coords: list) -> None:
        for point in coords:
            print(f'{point=}: {image[point[0], point[1]]}')

class TorchUtils:
    @staticmethod
    def warp_image(
        image: npt.NDArray[np.float32], motion_vectors: npt.NDArray[np.float32], change_direction: bool = True, align_corners: bool = False
    ) -> npt.NDArray[np.float32]:
        
        height, width = image.shape[:2]

        input_tensor = TorchUtils._image_to_tensor(image)

        base_grid = TorchUtils._get_base_grid(height, width)
        motion_grid = base_grid + TorchUtils._mv_to_grid(motion_vectors, change_direction, align_corners)

        warped_tensor = F.grid_sample(
            input_tensor, motion_grid,
            mode='bilinear',
            padding_mode="zeros",
            align_corners=align_corners
        )

        warped_image = TorchUtils._tensor_to_numpy(warped_tensor)

        return warped_image
    
    @staticmethod
    def _mv_to_grid(
        motion_vectors: npt.NDArray[np.float32], change_direction: bool, align_corners: bool
    ) -> npt.NDArray[np.float32]:
    
        motion_vectors = motion_vectors[..., :2]
        motion_vectors = motion_vectors[..., ::-1]

        height, width = motion_vectors.shape[:2]

        result = np.empty_like(motion_vectors, dtype=np.float32)
        result[..., 0] = motion_vectors[..., 0] * height
        result[..., 1] = motion_vectors[..., 1] * width
        
        corner = 1 if align_corners else 0
        result[..., 0] = result[..., 0] * (2.0 / (height - corner))
        result[..., 1] = result[..., 1] * (2.0 / (width - corner))

        sign = -1 if change_direction else 1
        return result * sign

    @staticmethod
    def _norm_to_mv(
        grid: npt.NDArray[np.float32], is_moving_backward: bool
    ) -> npt.NDArray[np.float32]:

        height, width = grid.shape[:2]

        sign = -1 if is_moving_backward else 1
        
        result = np.empty_like(grid, dtype=np.float32)
        result[..., 0] = grid[..., 0] * ((height - 1) / 2.0)
        result[..., 1] = grid[..., 1] * ((width - 1) / 2.0)

        return result * sign
    
    @staticmethod
    def _tensor_to_numpy(img: torch.Tensor) -> npt.NDArray[np.float32]:
        return img.detach().squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()

    @staticmethod
    def _get_base_grid(height: int, width: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device='cpu'),
            torch.linspace(-1, 1, width, device='cpu'),
            indexing='ij'
        )

        return torch.stack((xx, yy), dim=-1).unsqueeze(0).contiguous()
    
    @staticmethod
    def _image_to_tensor(img: npt.NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).clone().contiguous()

class ImageUtils:
    @staticmethod
    def normalize(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float16]:
        min = image.min()
        max = image.max()

        normalized = (image - min) / (max - min)
        return normalized
    
    @staticmethod
    def make_8bit(image: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        min = image.min()
        max = image.max()

        norm_image = (image - min) / (max - min)
        norm_image = (norm_image * 255.0)

        image_8bit = np.sum(norm_image, axis=-1) / image.shape[-1]
        
        return image_8bit.astype(np.uint8)
    
    @staticmethod
    def make_grayscale(img: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
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

        return max(quads, key=lambda q: q[1])[0]

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
