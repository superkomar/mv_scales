import os
import argparse
from typing import NamedTuple, Tuple
from enum import Enum
import logging

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from mv_scales_compute import KeypointsApproach
from mv_scales_compute import GradientDescentApproach
from mv_scales_compute import utils, ImageUtils


APPROACH_LIST = ['keypoints', 'gradient', 'all']
LOG_LEVELS_LIST = ['INFO', 'DEBUG']


class AppResult(NamedTuple):
    Name: str
    Scale_x: float
    Scale_y: float

    def __str__(self) -> str:
        return f'Approach name: {result.Name}; Scale X: {result.Scale_x:.6f}; Scale Y: {result.Scale_y:.6f}'
 
class Dataset(NamedTuple):
    Frame_1: str
    Frame_2: str
    Mv_1: str
    Mv_2: str


def get_defaults_dataset(name: str) -> Dataset:
    datasets = {
        'toyshop': Dataset(
            Frame_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'toyshop_00000.exr'),
            Frame_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'toyshop_00001.exr'),
            Mv_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'toyshop_00000.exr'),
            Mv_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'toyshop_00001.exr')
        ),

        'urban_city': Dataset(
            Frame_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'urban_city_02013.exr'),
            Frame_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'urban_city_02014.exr'),
            Mv_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'urban_city_02013.exr'),
            Mv_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'urban_city_02014.exr')
        ),

        'custom': Dataset(
            Frame_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'custom_0.exr'),
            Frame_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'custom_1.exr'),
            Mv_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'custom_0.exr'),
            Mv_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'custom_1.exr')
        ),

        'mv_puzzles': Dataset(
            Frame_1='',
            Frame_2='',
            Mv_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'mv_puzzles_velocity_00018.exr'),
            Mv_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'mv_puzzles_velocity_00019.exr')
        )
    }

    return datasets[name]

def parse_cli_arguments() -> Tuple[str, Dataset]:

    parser = argparse.ArgumentParser()
    parser.add_argument('-app', help='Choose one of the following approaches', choices=APPROACH_LIST, default=APPROACH_LIST[2])
    parser.add_argument('-mv_1', help='File path for the first img with motion vectors', required=True)
    parser.add_argument('-mv_2', help='File path for the second img with motion vectors', required=True)
    parser.add_argument('-frame_1', help='File path for the first frame', default='')
    parser.add_argument('-frame_2', help='File path for the second frame', default='')
    parser.add_argument('-log_level', help='Logging level', choices=LOG_LEVELS_LIST, default=LOG_LEVELS_LIST[0])

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    dataset = Dataset(args.frame_1, args.frame_2, args.mv_1, args.mv_2)
    # dataset = get_defaults_dataset('toyshop')

    return args.app, dataset

if __name__ == '__main__':

    approach, dataset = parse_cli_arguments()

    logger = logging.getLogger(__name__)
    
    frame_1 = utils.read_exr(dataset.Frame_1) if dataset.Frame_1 else None
    frame_2 = utils.read_exr(dataset.Frame_2) if dataset.Frame_1 else None
    motion_vectors_1 = utils.read_exr(dataset.Mv_1)
    motion_vectors_2 = utils.read_exr(dataset.Mv_2)

    results = []
    if approach == APPROACH_LIST[0] or approach == APPROACH_LIST[2]:
        app = KeypointsApproach()

        if frame_1 is not None and frame_2 is not None:
            scale_x, scale_y = app.from_frames(frame_1, frame_2, motion_vectors_2)

            results.append(AppResult(Name='Keypoints (frames)', Scale_x=scale_x, Scale_y=scale_y))

        else:
            scale_x, scale_y = app.from_motion_vectors(motion_vectors_1, motion_vectors_2)
            
            results.append(AppResult(Name='Keypoints (motion vectors)', Scale_x=scale_x, Scale_y=scale_y))

    if approach == APPROACH_LIST[1] or approach == APPROACH_LIST[2]:
        app = GradientDescentApproach()

        if frame_1 is not None and frame_2 is not None:
            scale_x, scale_y = app.from_frames(frame_1, frame_2, motion_vectors_2)

            results.append(AppResult(Name='Gradient descent (frames)', Scale_x=scale_x, Scale_y=scale_y))
        
        else:
            scale_x, scale_y = app.from_motion_vectors(motion_vectors_1, motion_vectors_2)
            
            results.append(AppResult(Name='Gradient descent (motion vectors)', Scale_x=scale_x, Scale_y=scale_y))

    for result in results:
        logger.info(f'{result}')

