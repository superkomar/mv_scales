import numpy as np
import numpy.typing as npt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn.parameter as pr
import logging
import os
from dataclasses import dataclass

from .utils import ExrUtils, ImageUtils
from .approach_base import ApproachBase, ApproachParameters


@dataclass
class GDParameters(ApproachParameters):
    StepsNum: int = 10000
    LearningRate: float = 1e-4
    IsMovingBackward: bool = True
    # ZeroEpsilon: float = 1e-6


class GradientDescent(ApproachBase):

    def __init__(self, parameters: GDParameters = GDParameters()) -> None:
        super().__init__()
        
        self._steps_num = parameters.StepsNum
        self._learning_rate = parameters.LearningRate
        self._is_moving_backward = parameters.IsMovingBackward
        self._zero_epsilon = parameters.ZeroEpsilon

        self.logger = logging.getLogger(__name__)
        self._is_debug = self.logger.getEffectiveLevel() == 10

        self._torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def from_motion_vectors(
            self, motion_vectors_1: npt.NDArray[np.float32], motion_vectors_2: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:
        
        if motion_vectors_1 is None or motion_vectors_2 is None:
            raise RuntimeError('One of the argument is None')

        if self._is_moving_backward:
            self.logger.debug('from "motion_vectors_2" to "motion_vectors_1"')

            source, target = motion_vectors_2, motion_vectors_1
            motion_vectors = motion_vectors_2[..., :2]
        
        else:
            self.logger.debug('from "motion_vectors_1" to "motion_vectors_2"')

            source, target = motion_vectors_1, motion_vectors_2
            motion_vectors = motion_vectors_1[..., :2]

        return self._compute_scales(source=source, target=target, motion_vectors=motion_vectors)
    
    def from_frames(
        self, frame_1: npt.NDArray[np.float32], frame_2: npt.NDArray[np.float32], motion_vectors: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:
        
        if frame_1 is None or frame_2 is None or motion_vectors is None:
            raise RuntimeError('One of the argument is None')
        
        if self._is_moving_backward:
            self.logger.debug('from "frame_2" to "frame_1"')

            source, target = frame_2, frame_1
        else:
            self.logger.debug('from "frame_1" to "frame_2"')

            source, target = frame_1, frame_2

        motion_vectors = motion_vectors[..., :2]

        return self._compute_scales(source=source, target=target, motion_vectors=motion_vectors)

    def _compute_scales(
        self, source: npt.NDArray[np.float32], target: npt.NDArray[np.float32], motion_vectors: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:
        
        motion_vectors = self._norm_to_grid(motion_vectors)

        source_tensor = self._numpy_to_tensor(source)
        target_tensor = self._numpy_to_tensor(target)
        
        mv_tensor = self._mv_to_parameter(motion_vectors)

        warped_tensor, final_mv_tensor = self._run_gradient_descent(
            input=source_tensor, motion_vectors=mv_tensor, target=target_tensor
        )

        custom_motion_vectors = self._mv_to_numpy(final_mv_tensor)

        # Debug condition
        if self._is_debug:
            dir_name = 'debug'
            os.makedirs('debug', exist_ok=True)
            # ExrUtils.write_exr(source, os.path.join(dir_name, 'source.exr'))
            # ExrUtils.write_exr(target, os.path.join(dir_name, 'target.exr'))
            # ExrUtils.write_exr(motion_vectors, os.path.join(dir_name, 'mv_original.exr'))
            ExrUtils.write_exr(custom_motion_vectors, os.path.join(dir_name, 'mv_final.exr'))
            ExrUtils.write_exr(self._tensor_to_numpy(warped_tensor), os.path.join(dir_name, 'warped_final.exr'))

        motion_vectors = self._norm_to_mv(motion_vectors)
        custom_motion_vectors = self._norm_to_mv(custom_motion_vectors)

        scale_x, scale_y = self.calculate_mv_scales(custom_motion_vectors, motion_vectors, self._zero_epsilon)
        
        return scale_x, scale_y
    
    def _run_gradient_descent(
        self,
        input: torch.Tensor,
        motion_vectors: pr.Parameter,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        height = input.shape[2]
        width  = input.shape[3]

        input = input.to(self._torch_device)
        target = target.to(self._torch_device)
        
        motion_vectors.data = motion_vectors.data.to(self._torch_device)

        base_grid = self._get_base_grid(height, width).to(self._torch_device)

        optimizer = torch.optim.AdamW([motion_vectors], lr=self._learning_rate)

        num_steps_log = self._steps_num * 0.1

        for step in range(self._steps_num):
            optimizer.zero_grad()

            motion_grid = base_grid + motion_vectors

            motion_grid = torch.clamp(motion_grid, -1.2, 1.2)

            # input [1, C, H, W]; grid [1, H, W, 2]
            warped_input = F.grid_sample(
                input, motion_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )

            loss = self._calc_loss(warped_input=warped_input, target=target)

            loss.backward()
            optimizer.step()

            if step % num_steps_log == 0:
                self.logger.debug(f'Step {step}: loss = {loss.item():.8f}')

        return warped_input, motion_vectors
    
    @staticmethod
    def _calc_loss(warped_input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # return F.mse_loss(warped_input, target)
        loss = F.smooth_l1_loss(warped_input, target)

        return loss
    
    @staticmethod
    def _get_base_grid(height: int, width: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device='cpu'),
            torch.linspace(-1, 1, width, device='cpu'),
            indexing='ij'
        )

        return torch.stack((xx, yy), dim=-1).unsqueeze(0).contiguous()
    
    @staticmethod
    def _norm_to_grid(motion_vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    
        motion_vectors = motion_vectors[..., :2]
        motion_vectors = motion_vectors[..., ::-1]

        height, width = motion_vectors.shape[:2]

        result = np.empty_like(motion_vectors, dtype=np.float32)
        result[..., 0] = motion_vectors[..., 0] * (width - 1)
        result[..., 1] = motion_vectors[..., 1] * (height - 1)

        result[..., 0] = result[..., 0] * (2.0 / (width - 1))
        result[..., 1] = result[..., 1] * (2.0 / (height - 1))

        return result

    @staticmethod
    def _norm_to_mv(grid: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:

        height, width = grid.shape[:2]
        
        result = np.empty_like(grid, dtype=np.float32)
        result[..., 0] = grid[..., 0] * ((width - 1) / 2.0)
        result[..., 1] = grid[..., 1] * ((height - 1) / 2.0)

        return result

    
    @staticmethod
    def _numpy_to_tensor(img: npt.NDArray[np.float32]) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).clone().contiguous()

    @staticmethod
    def _tensor_to_numpy(img: torch.Tensor) -> npt.NDArray[np.float32]:
        return img.detach().squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    
    @staticmethod
    def _mv_to_parameter(mv: npt.NDArray[np.float32]) -> pr.Parameter:
        return pr.Parameter(torch.from_numpy(mv).unsqueeze(0).contiguous().clone())

    @staticmethod
    def _mv_to_numpy(mv: torch.Tensor) -> npt.NDArray[np.float32]:
        return mv.detach().squeeze(0).contiguous().cpu().numpy()

