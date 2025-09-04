from typing import Tuple
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ApproachParameters:
    ZeroEpsilon: float = 1e-5


class ApproachBase(ABC):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._is_debug = self.logger.getEffectiveLevel() == 10

    @abstractmethod
    def from_frames(
        self, frame_1: npt.NDArray[np.float32], frame_2: npt.NDArray[np.float32], motion_vectors: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:
        pass

    @abstractmethod
    def from_motion_vectors(
        self, motion_vectors_1: npt.NDArray[np.float32], motion_vectors_2: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:
        pass

    def calculate_mv_scales(
        self, custom_mv: npt.NDArray[np.float32], original_mv: npt.NDArray[np.float32], zero_eps: float
    ) -> Tuple[float, float]:
        """
        Calculate the scale between custom and original motion vectors as the mean along each axis

        :param custom_mv: custom motion vectors, XY-ordered
        :param original_mv: original motion vectors, XY-ordered
        :param zero_eps: if the value is less than 'eps', it is treated as zero
        :return: tuple contains 'scale_x' and 'scale_y'
        """
        
        if custom_mv.shape != original_mv.shape:
            raise RuntimeError('shapes of motion vectors should be the same')
        
        custom_mv_x = custom_mv[..., 0]
        custom_mv_y = custom_mv[..., 1]

        original_mv_x = original_mv[..., 0]
        original_mv_y = original_mv[..., 1]

        not_zero_mask_x = (abs(original_mv_x) > zero_eps) & (abs(custom_mv_x) > zero_eps)
        not_zero_mask_y = (abs(original_mv_y) > zero_eps) & (abs(custom_mv_y) > zero_eps)

        scale_x = np.mean(custom_mv_x[not_zero_mask_x] / original_mv_x[not_zero_mask_x], dtype=np.float32)
        scale_y = np.mean(custom_mv_y[not_zero_mask_y] / original_mv_y[not_zero_mask_y], dtype=np.float32)

        # if self._is_debug:
            
        #     for idx in range(custom_mv.shape[0]):

        #         if not not_zero_mask_x[idx] and not not_zero_mask_y[idx]:
        #             continue

        #         if not_zero_mask_x[idx] and not_zero_mask_y[idx]:
        #             self.logger.debug(f'custom   | y={custom_mv_y[idx]:0.6f}; x={custom_mv_x[idx]:0.6f}')
        #             self.logger.debug(f'original | y={original_mv_y[idx]:0.6f}; x={original_mv_x[idx]:0.6f}')
                
        #         elif not_zero_mask_x[idx]:
        #             self.logger.debug(f'custom   | x={custom_mv_x[idx]:0.6f}')
        #             self.logger.debug(f'original | x={original_mv_x[idx]:0.6f}')

        #         else:
        #             self.logger.debug(f'custom   | y={custom_mv_y[idx]:0.6f}')
        #             self.logger.debug(f'original | y={original_mv_y[idx]:0.6f}')


        #         self.logger.debug('')

        return scale_x, scale_y

