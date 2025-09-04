import os
from typing import NamedTuple, Tuple
import logging

import numpy as np
import numpy.typing as npt

from mv_scales_compute import ExrUtils, ImageUtils, TorchUtils
from mv_scales_compute import KeypointsApproach, KPParameters
from mv_scales_compute import GradientDescentApproach, GDParameters
from mv_scales_compute import ApproachBase


class Dataset(NamedTuple):
    FramePath_1: str
    FramePath_2: str
    MvPath_1: str
    MvPath_2: str

    FramePoints_1: npt.NDArray[np.int32]
    FramePoints_2: npt.NDArray[np.int32]

    MvPoints_1: npt.NDArray[np.int32]
    MvPoints_2: npt.NDArray[np.int32]

    Width: int
    Height: int

    def has_frames_points(self) -> bool:
        return len(self.FramePoints_1) > 0 and len(self.FramePoints_2) > 0
    
    def has_mv_points(self) -> bool:
        return len(self.MvPoints_1) > 0 and len(self.MvPoints_2) > 0

    def get_frames_vectors(self) -> npt.NDArray[np.float32]:
        return self._get_vectors(end=self.FramePoints_2, begin=self.FramePoints_1)
    
    def get_mv_vectors(self) -> npt.NDArray[np.float32]:
        return self._get_vectors(end=self.MvPoints_2, begin=self.MvPoints_1)
    
    def _get_vectors(self, end, begin) -> npt.NDArray[np.float32]:
        mv = np.zeros((end.shape[0], 2))
        mv[..., 1] = (end[..., 1] - begin[..., 1]) / self.Width
        mv[..., 0] = (end[..., 0] - begin[..., 0]) / self.Height

        return mv


TestDataset = {
    'toyshop': Dataset(
        Width=1920, Height=1080,
        FramePath_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'toyshop_00002.exr'),
        FramePath_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'toyshop_00003.exr'),
        MvPath_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'toyshop_00002.exr'),
        MvPath_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'toyshop_00003.exr'),
        
        FramePoints_1=np.array([(659, 674), (882, 356), (909, 26), (100, 1766), (583, 899)]),
        FramePoints_2=np.array([(660, 675), (881, 355), (910, 21), ( 99, 1767), (582, 898)]),

        MvPoints_1=np.array([(649, 659), (834, 307), (1007, 38), (348, 1773), (739, 503)]),
        MvPoints_2=np.array([(650, 660), (833, 304), (1009, 32), (347, 1776), (740, 501)]),

        # FramePoints_1=[],
        # FramePoints_2=[],

        # MvPoints_1=[],
        # MvPoints_2=[],
    ),

    'urban_city': Dataset(
        Width=1920, Height=1080,
        FramePath_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'urban_city_02013.exr'),
        FramePath_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'urban_city_02014.exr'),
        MvPath_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'urban_city_02013.exr'),
        MvPath_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'urban_city_02014.exr'),

        FramePoints_1=[],
        FramePoints_2=[],

        MvPoints_1=[],
        MvPoints_2=[],
    ),

    'mv_puzzles': Dataset(
        Width=1920, Height=1080,
        FramePath_1='',
        FramePath_2='',
        MvPath_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'mv_puzzles_velocity_00018.exr'),
        MvPath_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'mv_puzzles_velocity_00019.exr'),
        
        FramePoints_1=[],
        FramePoints_2=[],

        MvPoints_1=[],
        MvPoints_2=[],
    ),

    'custom': Dataset(
        Width=1920, Height=1080,
        FramePath_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'custom_1.exr'),
        FramePath_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'custom_2.exr'),
        MvPath_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'custom_1.exr'),
        MvPath_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'custom_2.exr'),

        FramePoints_1=np.array([(100, 100), (100, 299), (399, 100), (399, 299), (200, 1500), (400, 1500)]),
        FramePoints_2=np.array([(400, 400), (400, 599), (699, 400), (699, 599), (200, 1000), (400, 1000)]),

        MvPoints_1=[],
        MvPoints_2=[],
    ),
}


def get_mv_values(mv: npt.NDArray[np.float32], coords: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
    return mv[coords[..., 0], coords[..., 1]]

def test_frames_approach(dataset: Dataset, logger: logging.Logger) -> None:
    if not (dataset.FramePath_1 and dataset.FramePath_2 and dataset.has_frames_points()):
        return

    logger.info('Frames comparison')

    zero_eps = 1e-4
    
    frame_1 = ExrUtils.read_exr(dataset.FramePath_1)
    frame_2 = ExrUtils.read_exr(dataset.FramePath_2)
    mv = ExrUtils.read_exr(dataset.MvPath_2)

    custom_mv = dataset.get_frames_vectors()
    original_mv = get_mv_values(mv, dataset.FramePoints_2)

    scale_x, scale_y = KeypointsApproach().calculate_mv_scales(
        custom_mv=custom_mv,
        original_mv=original_mv,
        zero_eps=zero_eps
    )
    logger.info(f'Manual    | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

    app = KeypointsApproach(KPParameters(MatchesThreshold=5, DetectAlgorithm='sift'))
    scale_x, scale_y = app.from_frames(frame_1, frame_2, mv)
    logger.info(f'Keypoints | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

    scale_x, scale_y = GradientDescentApproach().from_frames(frame_1, frame_2, mv)
    logger.info(f'Gradient  | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

    logger.info('')

def test_mv_approach(dataset: Dataset, logger: logging.Logger) -> None:
    if not (dataset.MvPath_1 and dataset.MvPath_2 and dataset.has_mv_points()):
        return

    logger.info('Motion vectors comparison')

    zero_eps = 1e-4
    
    mv_1 = ExrUtils.read_exr(dataset.MvPath_1)
    mv_2 = ExrUtils.read_exr(dataset.MvPath_2)

    custom_mv = dataset.get_mv_vectors()

    scale_x, scale_y = KeypointsApproach().calculate_mv_scales(
        custom_mv=custom_mv,
        original_mv=dataset.get_mv_vectors(),
        zero_eps=zero_eps
    )
    logger.info(f'Manual    | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

    app = KeypointsApproach(KPParameters(MatchesThreshold=25))
    scale_x, scale_y = app.from_motion_vectors(mv_1, mv_2)
    logger.info(f'Keypoints | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

    scale_x, scale_y = GradientDescentApproach().from_motion_vectors(mv_1, mv_2)
    logger.info(f'Gradient  | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

    logger.info()

if __name__ == '__main__':
    logging.basicConfig(
        level='DEBUG',
        format='%(levelname)s | %(message)s',
    )
    
    logger = logging.getLogger(__name__)

    for name, dataset in TestDataset.items():
        logger.info(f'\t"{name}" dataset')

        test_frames_approach(dataset, logger)
        test_mv_approach(dataset, logger)

