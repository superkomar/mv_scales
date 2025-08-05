import numpy as np
import numpy.typing as npt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn.parameter as pr
import logging

from .utils import read_exr, write_exr, ImageUtils


class GradientDescentApproach():
    _DEFAULT_STEPS_NUM_ = 500
    _DEFAULT_LEARNING_RATE_ = 1e-3

    def __init__(self,
                 steps_num: int = _DEFAULT_STEPS_NUM_,
                 learning_rate: float = _DEFAULT_LEARNING_RATE_,
                 is_moving_backward: bool = True
    ) -> None:
        self._steps_num = steps_num
        self._learning_rate = learning_rate
        self._is_moving_backward = is_moving_backward

        self.logger = logging.getLogger(__name__)
    
    def compute_from_frames(self,
        frame_1: npt.NDArray[np.float16], frame_2: npt.NDArray[np.float16], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        self.logger.info(f'Gradient Descent (frames) has started')

        move_sign = -1 if self._is_moving_backward else 1
        motion_vectors = motion_vectors[..., :2] * move_sign
        
        img_1_tensor = self._numpy_to_tensor(frame_1)
        img_2_tensor = self._numpy_to_tensor(frame_2)
        mv_tensor = self._mv_to_parameter(motion_vectors)

        warped_img = self._run_gradient_descent(img_2_tensor, mv_tensor, img_1_tensor)

        # Debug condition
        if self.logger.getEffectiveLevel() == 10:
            write_exr(frame_1, f'debug\\frame_1.exr')
            write_exr(frame_2, f'debug\\frame_2.exr')
            write_exr(motion_vectors, f'debug\\mv_original.exr')
            write_exr(self._mv_to_numpy(mv_tensor), f'debug\\mv_final.exr')
            write_exr(self._tensor_to_numpy(warped_img), f'debug\\warped_final.exr')

        custom_motion_vectors = self._mv_to_numpy(mv_tensor)

        scale_x, scale_y = self._calc_scales(custom=custom_motion_vectors, original=motion_vectors)

        self.logger.info(f'Gradient Descent (frames) has ended')

        return scale_x, scale_y
    
    
    def compute_from_motion_vectors(self, mv_1: npt.NDArray[np.float16], mv_2: npt.NDArray[np.float16]) -> Tuple[float, float]:
        if self._is_moving_backward:
            source, target = mv_2, mv_1
            motion_vectors = mv_2[..., :2] * -1
        else:
            source, target = mv_1, mv_2
            motion_vectors = mv_1[..., :2]
        
        return self._from_motion_vectors(source, target, motion_vectors)

    
    def _from_motion_vectors(
        self, source: npt.NDArray[np.float32], target: npt.NDArray[np.float32], motion_vectors: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:
        self.logger.info(f'Gradient Descent (motion vectors) has started')

        source_tensor = self._numpy_to_tensor(source)
        target_tensor = self._numpy_to_tensor(target)
        
        mv_tensor = self._mv_to_parameter(motion_vectors)

        warped_img = self._run_gradient_descent(source_tensor, mv_tensor, target_tensor)

        # Debug condition
        if self.logger.getEffectiveLevel() == 10:  
            write_exr(source, f'debug\\source.exr')
            write_exr(target, f'debug\\target.exr')
            write_exr(motion_vectors, f'debug\\mv_original.exr')
            write_exr(self._mv_to_numpy(mv_tensor), f'debug\\mv_final.exr')
            write_exr(self._tensor_to_numpy(warped_img), f'debug\\warped_final.exr')

        custom_motion_vectors = self._mv_to_numpy(mv_tensor)

        scale_x, scale_y = self._calc_scales(custom_motion_vectors, motion_vectors)
        
        self.logger.info(f'Gradient Descent (motion vectors) has ended')
        
        return scale_x, scale_y
    
    def _run_gradient_descent(self, input: torch.Tensor, motion_vectors: pr.Parameter, target: torch.Tensor):
        height = input.shape[2]
        width = input.shape[3]

        base_grid = self._get_base_grid(height, width)
        optimizer = torch.optim.SGD([motion_vectors], lr=self._learning_rate)

        for step in range(self._steps_num):
            optimizer.zero_grad()

            motion_grid = base_grid + motion_vectors

            # input [1, C, H, W]; grid [1, H, W, 2]
            warped_input = F.grid_sample(input, motion_grid, mode='bilinear', padding_mode="zeros", align_corners=True)

            loss = F.mse_loss(warped_input, target)

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                self.logger.debug(f'Step {step}: loss = {loss.item():.6f}')

        return warped_input
    
    @staticmethod
    def _get_base_grid(height: int, width: int) -> torch.Tensor:
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
    def _numpy_to_tensor(img: npt.NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).contiguous()

    @staticmethod
    def _tensor_to_numpy(img: torch.Tensor) -> npt.NDArray[np.float32]:
        return img.detach().squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    
    @staticmethod
    def _mv_to_parameter(mv: npt.NDArray[np.float32]) -> pr.Parameter:
        return pr.Parameter(torch.from_numpy(mv.astype(np.float32)).unsqueeze(0).contiguous().clone().detach())

    @staticmethod
    def _mv_to_numpy(mv: torch.Tensor) -> npt.NDArray[np.float32]:
        return mv.detach().squeeze(0).contiguous().cpu().numpy()

