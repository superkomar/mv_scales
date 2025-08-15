import numpy as np
import numpy.typing as npt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn.parameter as pr
import logging
import os
from dataclasses import dataclass

from .utils import read_exr, write_exr, ImageUtils

@dataclass
class ApproachParameters:
    StepsNum: int = 1000
    LearningRate: float = 1e1
    IsMovingBackward: bool = True


class GradientDescentApproach:

    def __init__(self, parameters: ApproachParameters = ApproachParameters()) -> None:
        
        self._steps_num = parameters.StepsNum
        self._learning_rate = parameters.LearningRate
        self._is_moving_backward = parameters.IsMovingBackward

        self.logger = logging.getLogger(__name__)
        self._is_debug = self.logger.getEffectiveLevel() == 10

        self.logger.debug(f'{torch.cuda.is_available()=}')
        self._torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def from_motion_vectors(
            self, motion_vectors_1: npt.NDArray[np.float16], motion_vectors_2: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        if motion_vectors_1 is None or motion_vectors_2 is None:
            raise RuntimeError('One of the argument is None')

        if self._is_moving_backward:
            source, target = motion_vectors_2, motion_vectors_1
            motion_vectors = motion_vectors_2[..., :2] * (-1)

            self.logger.debug('from "motion_vectors_2" to "motion_vectors_1"')
        else:
            source, target = motion_vectors_1, motion_vectors_2
            motion_vectors = motion_vectors_1[..., :2]

            self.logger.debug('from "motion_vectors_1" to "motion_vectors_2"')

        return self._compute_scales(source=source, target=target, motion_vectors=motion_vectors)
    
    def from_frames(
        self, frame_1: npt.NDArray[np.float16], frame_2: npt.NDArray[np.float16], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        if frame_1 is None or frame_2 is None or motion_vectors is None:
            raise RuntimeError('One of the argument is None')
        
        if self._is_moving_backward:
            source, target = frame_2, frame_1
            motion_vectors = motion_vectors[..., :2] * (-1)
            
            self.logger.debug('from "frame_2" to "frame_1"')
        else:
            source, target = frame_1, frame_2
            motion_vectors = motion_vectors[..., :2]

            self.logger.debug('from "frame_1" to "frame_2"')

        return self._compute_scales(source=source, target=target, motion_vectors=motion_vectors)

    def _compute_scales(
        self, source: npt.NDArray[np.float32], target: npt.NDArray[np.float32], motion_vectors: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:
        
        self.logger.info(f'Gradient Descent has started')

        source_tensor = self._numpy_to_tensor(source)
        target_tensor = self._numpy_to_tensor(target)
        
        mv_tensor = self._mv_to_parameter(motion_vectors)

        # yy, xx = torch.meshgrid(
        #     torch.linspace(0, 1, source.shape[0]),
        #     torch.linspace(0, 1, source.shape[1]),
        #     indexing='ij'
        # )
        # mv_tensor = pr.Parameter(
        #     torch.rand((source.shape[0], source.shape[1], 2))
        #     .unsqueeze(0).contiguous().clone().detach()
        # )

        warped_img = self._run_gradient_descent(input=source_tensor, motion_vectors=mv_tensor, target=target_tensor)

        # Debug condition
        if self._is_debug:
            dir_name = 'debug'
            os.makedirs('debug', exist_ok=True)
            write_exr(source, os.path.join(dir_name, 'source.exr'))
            write_exr(target, os.path.join(dir_name, 'target.exr'))
            write_exr(motion_vectors, os.path.join(dir_name, 'mv_original.exr'))
            write_exr(self._mv_to_numpy(mv_tensor), os.path.join(dir_name, 'mv_final.exr'))
            write_exr(self._tensor_to_numpy(warped_img), os.path.join(dir_name, 'warped_final.exr'))

        custom_motion_vectors = self._mv_to_numpy(mv_tensor)

        scale_x, scale_y = self._calc_motion_vectors_scales(custom_motion_vectors, motion_vectors)
        
        self.logger.info(f'Gradient Descent has completed')
        
        return scale_x, scale_y
    
    def _run_gradient_descent(self, input: torch.Tensor, motion_vectors: pr.Parameter, target: torch.Tensor):
        height = input.shape[2]
        width = input.shape[3]

        input = input.to(self._torch_device)
        target = target.to(self._torch_device)
        motion_vectors = pr.Parameter(motion_vectors.to(self._torch_device))
        base_grid = self._get_base_grid(height, width).to(self._torch_device)

        optimizer = torch.optim.SGD([motion_vectors], lr=self._learning_rate, momentum=0.9)

        for step in range(self._steps_num):
            optimizer.zero_grad()

            motion_grid = base_grid + motion_vectors
            # motion_grid = motion_vectors

            # input [1, C, H, W]; grid [1, H, W, 2]
            warped_input = F.grid_sample(input, motion_grid, mode='bilinear', padding_mode="zeros", align_corners=True)

            loss = F.mse_loss(warped_input, target)

            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                self.logger.debug(f'Step {step}: loss = {loss.item():.8f}')

        return warped_input
    
    @staticmethod
    def _calc_motion_vectors_scales(
        custom: npt.NDArray[np.float32],
        original: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:

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
    def _get_base_grid(height: int, width: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )

        return torch.stack((xx, yy), dim=-1).unsqueeze(0)
    
    @staticmethod
    def _numpy_to_tensor(img: npt.NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).contiguous()

    @staticmethod
    def _tensor_to_numpy(img: torch.Tensor) -> npt.NDArray[np.float32]:
        return img.detach().squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    
    @staticmethod
    def _mv_to_parameter(mv: npt.NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(mv.astype(np.float32)).unsqueeze(0).contiguous().clone().detach()

    @staticmethod
    def _mv_to_numpy(mv: torch.Tensor) -> npt.NDArray[np.float32]:
        return mv.detach().squeeze(0).contiguous().cpu().numpy()

