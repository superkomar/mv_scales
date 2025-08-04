import numpy as np
import numpy.typing as npt
from typing import Tuple
import torch
import torch.nn.functional as F
import logging

from .utils import read_exr, write_exr, ImageUtils


class GradientDescentApproach():
    _DEFAULT_STEPS_NUM_ = 500
    _DEFAULT_LEARNING_RATE_ = 1e-3

    def __init__(self, steps_num: int = _DEFAULT_STEPS_NUM_, learning_rate: float = _DEFAULT_LEARNING_RATE_) -> None:
        self._steps_num = steps_num
        self._learning_rate = learning_rate

        self.logger = logging.getLogger(__name__)

    def compute_from_frames(self,
        frame_1: npt.NDArray[np.float16], frame_2: npt.NDArray[np.float16], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        self.logger.info(f'Gradient Descent (frames) has started')

        height, width = frame_1.shape[:2]

        motion_vectors = motion_vectors[..., :2]
        
        img_1_tensor = self._input_to_tensor(frame_1)
        img_2_tensor = self._input_to_tensor(frame_2)

        mv_tensor = self._gird_to_tensor(motion_vectors).clone().detach().requires_grad_()
        base_grid = self._construct_base_grid(height, width)

        optimizer = torch.optim.Adam([mv_tensor], lr=self._learning_rate)

        for step in range(self._steps_num):
            optimizer.zero_grad()

            motion_grid = base_grid + mv_tensor

            # input [1, C, H, W]; grid [1, H, W, 2]
            warped_img_tensor = F.grid_sample(img_1_tensor, motion_grid, mode='bilinear', padding_mode="zeros", align_corners=True)

            # loss = F.l1_loss(warped_img_tensor, img_2_tensor)
            loss = F.mse_loss(warped_img_tensor, img_2_tensor)

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                self.logger.debug(f'Step {step}: L1 loss = {loss.item():.6f}')

        # Debug condition
        if self.logger.getEffectiveLevel() == 10:
            write_exr(frame_1, f'debug\\frame_1.exr')
            write_exr(frame_2, f'debug\\frame_2.exr')
            write_exr(motion_vectors, f'debug\\mv_original.exr')
            write_exr(self._input_to_numpy(warped_img_tensor), f'debug\\warped_final.exr')
            write_exr(self._grid_to_numpy(mv_tensor), f'debug\\mv_final.exr')

        custom_motion_vectors = self._grid_to_numpy(mv_tensor)

        scale_x, scale_y = self._calc_scales(custom=custom_motion_vectors, original=motion_vectors)

        self.logger.info(f'Gradient Descent (frames) has ended')

        return scale_x, scale_y

    def compute_from_motion_vectors(self, mv_1: npt.NDArray[np.float16], mv_2: npt.NDArray[np.float16]) -> Tuple[float, float]:
        raise RuntimeError('not implemented yet')
    
    @staticmethod
    def _construct_base_grid(height: int, width: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )

        return torch.stack((xx, yy), dim=-1).unsqueeze(0)

    @staticmethod
    def _calc_scales(custom: npt.NDArray[np.float32], original: npt.NDArray[np.float32]) -> Tuple[float, float]:

        custom_x = custom[..., 1]
        custom_y = custom[..., 0]
        
        original_x = original[..., 1]
        original_y = original[..., 0]

        eps = 1e-6

        zero_mask_x = np.abs(original_x) > eps
        zero_mask_y = np.abs(original_y) > eps

        scale_x = np.mean(custom_x[zero_mask_x] / original_x[zero_mask_x], dtype=np.float32)
        scale_y = np.mean(custom_y[zero_mask_y] / original_y[zero_mask_y], dtype=np.float32)

        return scale_x, scale_y
    
    @staticmethod
    def warp_image(img: npt.NDArray[np.float32], mv: npt.NDArray[np.float32]):
        height, width = img.shape[:2]

        img = GradientDescentApproach._input_to_tensor(img)
        mv = GradientDescentApproach._gird_to_tensor(mv)
        base_grid = GradientDescentApproach._construct_base_grid(height, width)

        motion_grid = base_grid + mv

        # input [1, C, H, W]; grid [1, H, W, 2]
        warped_img_tensor = F.grid_sample(img, motion_grid, mode='bilinear', padding_mode="zeros", align_corners=True)

        return GradientDescentApproach._input_to_numpy(warped_img_tensor)
    
    @staticmethod
    def _input_to_tensor(img: npt.NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).contiguous()

    @staticmethod
    def _input_to_numpy(img: torch.Tensor) -> npt.NDArray[np.float32]:
        return img.detach().squeeze(0).permute(1,2,0).contiguous().cpu().numpy()

    @staticmethod
    def _gird_to_tensor(mv: npt.NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(mv.astype(np.float32)).unsqueeze(0).contiguous()

    @staticmethod
    def _grid_to_numpy(mv: torch.Tensor) -> npt.NDArray[np.float32]:
        return mv.detach().squeeze(0).contiguous().cpu().numpy()

