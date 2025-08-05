import os
import argparse
from typing import NamedTuple
import logging

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from mv_scales_compute import KeypointsApproach
from mv_scales_compute import GradientDescentApproach
from mv_scales_compute import utils, ImageUtils


APPROACH_LIST = ['keypoints', 'gradient', 'all']
LOG_LEVELS_LIST = ['DEBUG', 'INFO']


class AppResult(NamedTuple):
    Name: str
    Scale_x: float
    Scale_y: float

    def __str__(self) -> str:
        return f'Approach name: {result.Name}; Scale X: {result.Scale_x}; Scale Y: {result.Scale_y}'

 
class DefPaths(NamedTuple):
    Frame_1: str
    Frame_2: str
    Mv_1: str
    Mv_2: str


def get_defaults_dataset(name: str) -> DefPaths:
    datasets = {
        'toyshop': DefPaths(
            Frame_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'toyshop_00000.exr'),
            Frame_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'toyshop_00001.exr'),
            Mv_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'toyshop_00000.exr'),
            Mv_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'toyshop_00001.exr')
        ),

        'urban_city': DefPaths(
            Frame_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'urban_city_02013.exr'),
            Frame_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'urban_city_02014.exr'),
            Mv_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'urban_city_02013.exr'),
            Mv_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'urban_city_02014.exr')
        ),

        'custom': DefPaths(
            Frame_1=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'custom_0.exr'),
            Frame_2=os.path.join(os.path.dirname(__file__), 'examples', 'frames', 'custom_1.exr'),
            Mv_1=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'custom_0.exr'),
            Mv_2=os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', 'custom_1.exr')
        )
    }

    return datasets[name]


def parse_cli_arguments():
    def_dataset = get_defaults_dataset('toyshop')

    parser = argparse.ArgumentParser()
    parser.add_argument('-app', help='Choose one of the following approaches', choices=APPROACH_LIST, default=APPROACH_LIST[2])
    parser.add_argument('-frame_1', help='File path for the first frame', default=def_dataset.Frame_1)
    parser.add_argument('-frame_2', help='File path for the second frame', default=def_dataset.Frame_2)
    parser.add_argument('-mv_1', help='File path for the first img with motion vectors', default=def_dataset.Mv_1)
    parser.add_argument('-mv_2', help='File path for the second img with motion vectors', default=def_dataset.Mv_2)
    parser.add_argument('-log_level', help='Logging level', choices=LOG_LEVELS_LIST, default=LOG_LEVELS_LIST[0])

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_cli_arguments()

    logging.basicConfig(
        level=args.log_level.upper(),
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    logger = logging.getLogger(__name__)

    frame_1 = utils.read_exr(args.frame_1)
    frame_2 = utils.read_exr(args.frame_2)
    motion_vectors_1 = utils.read_exr(args.mv_1)
    motion_vectors_2 = utils.read_exr(args.mv_2)

    results = []
    if args.app == APPROACH_LIST[0] or args.app == APPROACH_LIST[2]:
        app = KeypointsApproach()

        scale_x, scale_y = app.compute_from_frames(frame_1, frame_2, motion_vectors_2)

        results.append(AppResult(
            Name='Keypoints (frames)',
            Scale_x=scale_x,
            Scale_y=scale_y
        ))

        # scale_x, scale_y = app.compute_from_motion_vectors(motion_vectors_1, motion_vectors_2)
        
        # results.append(AppResult(
        #     Name='Keypoints (motion vectors)',
        #     Scale_x=scale_x,
        #     Scale_y=scale_y
        # ))

    if args.app == APPROACH_LIST[1] or args.app == APPROACH_LIST[2]:
        app = GradientDescentApproach(steps_num=1000)

        scale_x, scale_y = app.compute_from_frames(frame_1, frame_2, motion_vectors_2)

        results.append(AppResult(
            Name='Gradient descent (frames)',
            Scale_x=scale_x,
            Scale_y=scale_y
        ))

        scale_x, scale_y = app.compute_from_motion_vectors_new(motion_vectors_1, motion_vectors_2)

        results.append(AppResult(
            Name='Gradient descent (motion vectors)',
            Scale_x=scale_x,
            Scale_y=scale_y
        ))

    naive_warp = ImageUtils.warp_image(frame_2, motion_vectors_2, is_moving_forward=False)
    utils.write_exr(naive_warp, 'debug\\naive_warp.exr')

    for result in results:
        logger.info(result)
