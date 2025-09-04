from typing import Tuple
from abc import ABC, abstractmethod
import logging

import numpy as np
import numpy.typing as npt

class ApproachBase(ABC):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._is_debug = self.logger.getEffectiveLevel() == 10

    @abstractmethod
    def from_frames(
        frame_1: npt.NDArray[np.float32], frame_2: npt.NDArray[np.float32], motion_vectors: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:
        pass

    @abstractmethod
    def from_motion_vectors(
        motion_vectors_1: npt.NDArray[np.float32], motion_vectors_2: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:
        pass

    def calculate_mv_scales(
        self, custom_mv: npt.NDArray[np.float32], original_mv: npt.NDArray[np.float32], zero_eps: float
    ) -> Tuple[float, float]:
        
        custom_mv_x = custom_mv[..., 1]
        custom_mv_y = custom_mv[..., 0]

        original_mv_x = original_mv[..., 1]
        original_mv_y = original_mv[..., 0]

        zero_mask_x = abs(original_mv_x) > zero_eps
        zero_mask_y = abs(original_mv_y) > zero_eps

        scale_x = np.mean(custom_mv_x[zero_mask_x] / original_mv_x[zero_mask_x])
        scale_y = np.mean(custom_mv_y[zero_mask_y] / original_mv_y[zero_mask_y])

        if self._is_debug:
            
            for idx in range(custom_mv.shape[0]):
                self.logger.debug(f'custom   | y={custom_mv_y[idx]:0.6f}; x={custom_mv_x[idx]:0.6f}')
                self.logger.debug(f'original | y={original_mv_y[idx]:0.6f}; x={original_mv_x[idx]:0.6f}')
                self.logger.debug('')

        return scale_x, scale_y

