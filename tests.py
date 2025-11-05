import os
from typing import NamedTuple, Callable
import logging
import argparse

import numpy as np
import numpy.typing as npt

from mv_scales_compute import ExrUtils, ImageUtils, TorchUtils
from mv_scales_compute import Keypoints, KPParameters
from mv_scales_compute import GradientDescent, GDParameters
from mv_scales_compute import ApproachBase, Method


TARGETS = ['all', 'frames', 'mv']
APPROACHES = ['all', 'kp', 'gd']


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

    Rotate_xy: bool

    def has_frames_points(self) -> bool:
        return len(self.FramePoints_1) > 0 and len(self.FramePoints_2) > 0
    
    def has_mv_points(self) -> bool:
        return len(self.MvPoints_1) > 0 and len(self.MvPoints_2) > 0

    def get_frames_vectors(self) -> npt.NDArray[np.float32]:
        return self._get_vectors(end=self.FramePoints_1, begin=self.FramePoints_2)
    
    def get_mv_vectors(self) -> npt.NDArray[np.float32]:
        return self._get_vectors(end=self.MvPoints_1, begin=self.MvPoints_2)
    
    def _get_vectors(self, end, begin) -> npt.NDArray[np.float32]:
        mv = np.zeros((end.shape[0], 2), dtype=np.float32)
        mv[..., 0] = (end[..., 1] - begin[..., 1]) / self.Width
        mv[..., 1] = (end[..., 0] - begin[..., 0]) / self.Height

        return mv


TEST_DATASET = {
    'toyshop': Dataset(
        Rotate_xy=True,
        Width=1920, Height=1080,
        FramePath_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'toyshop_00002.exr'),
        FramePath_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'toyshop_00003.exr'),
        MvPath_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'toyshop_00002.exr'),
        MvPath_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'toyshop_00003.exr'),
        
        FramePoints_1=np.array([(659, 674), (882, 356), (909, 26), (100, 1766), (583, 899)]),
        FramePoints_2=np.array([(660, 675), (881, 355), (910, 21), ( 99, 1767), (582, 898)]),

        MvPoints_1=np.array([(650, 659), (834, 307), (1007, 38), (348, 1773), (739, 503)]),
        MvPoints_2=np.array([(651, 660), (833, 304), (1009, 32), (347, 1776), (740, 501)]),
    ),

    'urban_city': Dataset(
        Rotate_xy=True,
        Width=1920, Height=1080,
        FramePath_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'urban_city_02013.exr'),
        FramePath_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'urban_city_02014.exr'),
        MvPath_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'urban_city_02013.exr'),
        MvPath_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'urban_city_02014.exr'),

        FramePoints_1=np.array([]),
        FramePoints_2=np.array([]),

        MvPoints_1=np.array([]),
        MvPoints_2=np.array([]),
    ),

    'mv_puzzles': Dataset(
        Rotate_xy=True,
        Width=1920, Height=1080,
        FramePath_1='',
        FramePath_2='',
        MvPath_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'mv_puzzles_00018.exr'),
        MvPath_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'mv_puzzles_00019.exr'),
        
        FramePoints_1=np.array([]),
        FramePoints_2=np.array([]),

        MvPoints_1=np.array([(237, 218), (265, 255), (139, 662), (137, 785), (428, 956), (342, 1157), (630, 180), (503, 313)]),
        MvPoints_2=np.array([(238, 219), (266, 256), (142, 663), (140, 786), (429, 955), (344, 1156), (628, 182), (502, 314)]),
    ),

    'custom': Dataset(
        Rotate_xy=False,
        Width=1920, Height=1080,
        FramePath_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'custom_1.exr'),
        FramePath_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'custom_2.exr'),
        MvPath_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'custom_1.exr'),
        MvPath_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'custom_2.exr'),

        FramePoints_1=np.array([(100, 100), (100, 299), (399, 100), (399, 299), (200, 1500), (400, 1500)]),
        FramePoints_2=np.array([(400, 400), (400, 599), (699, 400), (699, 599), (200, 1000), (400, 1000)]),

        MvPoints_1=np.array([(100, 100), (100, 299), (399, 100), (399, 299), (200, 1500), (400, 1500)]),
        MvPoints_2=np.array([(400, 400), (400, 599), (699, 400), (699, 599), (200, 1000), (400, 1000)]),
    ),
}


class TestLauncher:

    def __init__(self, dataset: Dataset, data_slicer: Callable, app_chooser: Callable, debug_dir: str):
        self.dataset = dataset
        self.data_slicer = data_slicer
        self.app_chooser = app_chooser
        self.rotate_xy = dataset.Rotate_xy
        self.debug_dir = debug_dir

        self.zero_eps = 1e-6

        self.logger = logging.getLogger(__name__)

    def test_frames_approach(self) -> None:
        if not (self.dataset.FramePath_1 and self.dataset.FramePath_2 and dataset.MvPath_2):
            return

        self.logger.info('! Frames comparison !')

        frame_1 = ExrUtils.read_image(dataset.FramePath_1)
        frame_2 = ExrUtils.read_image(dataset.FramePath_2)
        mv = ExrUtils.read_motion_vectors(dataset.MvPath_2, rotate_xy=self.rotate_xy)

        if self.dataset.has_frames_points():
            scale_x, scale_y = self.calc_manual(
                custom_mv=dataset.get_frames_vectors(),
                original_mv=self.get_mv_values(mv, dataset.FramePoints_2),
                zero_eps=self.zero_eps
            )
            self.logger.info(f'Manual    | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

        frame_1 = self.data_slicer(frame_1)
        frame_2 = self.data_slicer(frame_2)
        mv = self.data_slicer(mv)

        if self.app_chooser(APPROACHES[1]):
            app = Keypoints(KPParameters(ZeroEpsilon=self.zero_eps))
            scale_x, scale_y = app.from_frames(frame_1, frame_2, mv)
            self.logger.info(f'Keypoints | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

        if self.app_chooser(APPROACHES[2]):
            app = GradientDescent(GDParameters(
                ZeroEpsilon=self.zero_eps, DebugDir=self.debug_dir
            ))
            scale_x, scale_y = app.from_frames(frame_1, frame_2, mv)
            self.logger.info(f'Gradient  | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

        logger.info('')

    def test_mv_approach(self) -> None:
        if not (dataset.MvPath_1 and dataset.MvPath_2):
            return

        self.logger.info('! Motion vectors comparison !')

        mv_1 = ExrUtils.read_motion_vectors(dataset.MvPath_1, self.rotate_xy)
        mv_2 = ExrUtils.read_motion_vectors(dataset.MvPath_2, self.rotate_xy)

        if dataset.has_mv_points():
            scale_x, scale_y = self.calc_manual(
                custom_mv=dataset.get_mv_vectors(),
                original_mv=self.get_mv_values(mv_2, dataset.MvPoints_2),
                zero_eps=self.zero_eps
            )
            self.logger.info(f'Manual    | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

        mv_1 = self.data_slicer(mv_1)
        mv_2 = self.data_slicer(mv_2)

        if self.app_chooser(APPROACHES[1]):
            app = Keypoints(KPParameters(
                ZeroEpsilon=self.zero_eps
            ))
            
            scale_x, scale_y = app.from_motion_vectors(mv_1, mv_2)
            self.logger.info(f'Keypoints | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

        if self.app_chooser(APPROACHES[2]):
            app = GradientDescent(GDParameters(
                ZeroEpsilon=self.zero_eps, DebugDir=self.debug_dir
            ))

            scale_x, scale_y = app.from_motion_vectors(mv_1, mv_2)
            self.logger.info(f'Gradient  | scale_x = {scale_x:.8f}; scale_y = {scale_y:.8f}')

        self.logger.info('')

    @staticmethod
    def calc_manual(custom_mv: npt.NDArray[np.float32], original_mv: npt.NDArray[np.float32], zero_eps: float):
        return Keypoints().calculate_scales(
            custom_mv=custom_mv,
            original_mv=original_mv,
            zero_eps=zero_eps,
            method=Method.median
        )

    @staticmethod
    def get_mv_values(mv: npt.NDArray[np.float32], coords: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        return mv[coords[..., 0], coords[..., 1]]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-target', choices=TARGETS, default=TARGETS[0])
    parser.add_argument('-app', choices=APPROACHES, default=APPROACHES[0])
    parser.add_argument('-log', choices=['info', 'debug'], default='debug')
    parser.add_argument('-slice', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log.upper(),
        format='%(levelname)s | %(message)s',
    )
    
    logger = logging.getLogger(__name__)

    dataset_names = [
        # 'custom',
        'toyshop',
        # 'urban_city',
        # 'mv_puzzles',
    ]

    app_chooser = lambda name: args.app == APPROACHES[0] or args.app == name

    if args.slice: 
        # slice_y = slice(500, 1000)
        # slice_x = slice(200,  800)
        slice_y = slice(0, 500)
        slice_x = slice(0, 500)

        data_slicer = lambda img: img[slice_y, slice_x]
    else:
        data_slicer = lambda img: img

    debug_dir = os.path.join(os.path.dirname(__file__), 'debug')

    for name in dataset_names:
        logger.info('')
        logger.info(f'=== {name.upper()} ===')

        dataset = TEST_DATASET[name]

        tests = TestLauncher(
            dataset=dataset,
            app_chooser=app_chooser,
            data_slicer=data_slicer,
            debug_dir=os.path.join(debug_dir, name)
        )

        if args.target == TARGETS[0] or args.target == TARGETS[1]:
            tests.test_frames_approach()
        
        if args.target == TARGETS[0] or args.target == TARGETS[2]:
            tests.test_mv_approach()
