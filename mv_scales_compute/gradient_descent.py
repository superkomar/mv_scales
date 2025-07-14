import numpy as np
import numpy.typing as npt
from typing import Tuple

from .utils import read_exr

class GradientDescentAlgorithm():
    @staticmethod
    def compute_from_matrices(
        frame_1: npt.NDArray[np.float16], frame_2: npt.NDArray[np.float16], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        raise RuntimeError('not implemented yet')
    
    @staticmethod
    def compute_from_files(frame_1: str, frame_2: str, motion_vectors: str) -> Tuple[float, float]:
        return GradientDescentAlgorithm.compute_from_matrices(
            frame_1=read_exr(frame_1),
            frame_2=read_exr(frame_2),
            motion_vectors=read_exr(motion_vectors)
        )

