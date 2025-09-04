import argparse
from typing import NamedTuple, Tuple
import logging

from mv_scales_compute import ApproachBase
from mv_scales_compute import KeypointsApproach
from mv_scales_compute import GradientDescentApproach
from mv_scales_compute import ExrUtils


APPROACH_LIST = ['keypoints', 'gradient']
LOG_LEVELS_LIST = ['INFO', 'DEBUG']


class Dataset(NamedTuple):
    Frame_1: str
    Frame_2: str
    Mv_1: str
    Mv_2: str


def parse_cli_arguments() -> Tuple[str, Dataset]:

    parser = argparse.ArgumentParser()
    parser.add_argument('-app', help='Choose one of the following approaches', choices=APPROACH_LIST, default=APPROACH_LIST[2])
    parser.add_argument('-mv_1', help='File path for the first img with motion vectors', required=True)
    parser.add_argument('-mv_2', help='File path for the second img with motion vectors', required=True)
    parser.add_argument('-frame_1', help='File path for the first frame', default='')
    parser.add_argument('-frame_2', help='File path for the second frame', default='')
    parser.add_argument('-log_level', help='Logging level', choices=LOG_LEVELS_LIST, default=LOG_LEVELS_LIST[0])

    args = parser.parse_args()
    
    dataset = Dataset(args.frame_1, args.frame_2, args.mv_1, args.mv_2)

    logging.basicConfig(
        level=args.log_level.upper(),
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    return args.app, dataset

def get_approach(name: str) -> ApproachBase:
    if name == APPROACH_LIST[0]:
        return KeypointsApproach()
    
    elif name == APPROACH_LIST[1]:
        return GradientDescentApproach()
    
    else:
        raise RuntimeError(f'{name} is incorrect approach name')

if __name__ == '__main__':

    approach_name, dataset = parse_cli_arguments()

    logger = logging.getLogger(__name__)

    approach = get_approach(approach_name)

    if dataset.Frame_1 is not '' and dataset.Frame_2 is not '':
        frame_1 = ExrUtils.read_exr(dataset.Frame_1)
        frame_2 = ExrUtils.read_exr(dataset.Frame_2)
        motion_vectors = ExrUtils.read_exr(dataset.Mv_2)
        
        scale_x, scale_y = approach.from_frames(frame_1, frame_2, motion_vectors)

    else:
        motion_vectors_1 = ExrUtils.read_exr(dataset.Mv_1)
        motion_vectors_2 = ExrUtils.read_exr(dataset.Mv_2)

        scale_x, scale_y = approach.from_motion_vectors(motion_vectors_1, motion_vectors_2)

    logger.info(f'Scale X: {scale_x:.6f}; Scale Y: {scale_y:.6f}')
