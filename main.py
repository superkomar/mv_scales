import argparse
import os
from typing import NamedTuple, Iterator, Tuple
import logging
from enum import Enum

from mv_scales_compute import __version__ as module_version
from mv_scales_compute import ApproachBase, Method
from mv_scales_compute import Keypoints, KPParameters
from mv_scales_compute import GradientDescent, GDParameters
from mv_scales_compute import ExrUtils


APPROACH_LIST = ['gradient', 'keypoints']
LOG_LEVELS_LIST = ['info', 'debug']

METHOD_LIST = [x.name for x in Method]
METHOD_DEF = METHOD_LIST[0]


class DatasetType(Enum):
    MotionVectors = 1
    Frames = 2


class Dataset(NamedTuple):
    Frame_1: str
    Frame_2: str
    MV_1: str
    MV_2: str

    def __str__(self):
        if self.Frame_1 != '' and self.Frame_2 != '':
            return f'Frame_1: {self.Frame_1}; Frame_2: {self.Frame_2}; MV: {self.MV_2}'
        
        return f'MV_1: {self.MV_1}; MV_2: {self.MV_2}'
    
    def validate(self) -> Tuple[bool, DatasetType]:
        is_valid_frame_1 = self.is_filepath_valid(self.Frame_1)
        is_valid_frame_2 = self.is_filepath_valid(self.Frame_2)

        is_valid_mv_1 = self.is_filepath_valid(self.MV_1)
        is_valid_mv_2 = self.is_filepath_valid(self.MV_2)

        dataset_type = DatasetType.MotionVectors
        is_dataset_valid = is_valid_mv_1 and is_valid_mv_2
        
        if is_valid_frame_1 and is_valid_frame_2 and is_valid_mv_2:
            dataset_type = DatasetType.Frames
            is_dataset_valid = True
            
        return is_dataset_valid, dataset_type
    
    @staticmethod
    def is_filepath_valid(path: str) -> bool:
        return os.path.exists(path) and os.path.isfile(path)


class ResultScales(NamedTuple):
    ScaleX: float
    ScaleY: float
    SrcInfo: str
    Message: str = ''

    def __str__(self) -> str:
        head = 'Result'
        footer = f'Source: {self.SrcInfo}'

        body = f'Scale X: {self.ScaleX}; Scale Y: {self.ScaleY}'
        if self.Message != '':
            body = f'Message: {self.Message}'

        return f'{head} | {body} | {footer}'


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    runtime_group = parser.add_argument_group(title='Runtime arguments')
    runtime_group.add_argument(
        '-app',
        help='the approach to be used',
        choices=APPROACH_LIST,
        default=APPROACH_LIST[0]
    )
    runtime_group.add_argument(
        '-method',
        help='the method to compute the scales',
        choices=METHOD_LIST,
        default=METHOD_DEF
    )
    runtime_group.add_argument(
        '-log_level',
        help='logging level',
        choices=LOG_LEVELS_LIST,
        default=LOG_LEVELS_LIST[0]
    )
    runtime_group.add_argument(
        '-log_to_file',
        help='flag to redirect logs to a file',
        action='store_true',
        default=False
    )

    # sub-parser to split arguments for files or whole folder
    subparser = parser.add_subparsers(
        title='Input data source',
        dest='input_data',
        description='choose which kind of input data will be used',
        help='kinds of input',
        required=True
    )

    # sub-command for files
    files_parser = subparser.add_parser('files', help='work with files only')
    files_parser.add_argument('-mv_1', help='File path for the first img with motion vectors', required=True)
    files_parser.add_argument('-mv_2', help='File path for the second img with motion vectors', required=True)
    files_parser.add_argument('-frame_1', help='File path for the first frame', default='')
    files_parser.add_argument('-frame_2', help='File path for the second frame', default='')

    # sub-command for folder
    folder_parser = subparser.add_parser('folders', help='work with files from folders')
    folder_parser.add_argument('-mv', help='Folder path for motion vector images', required=True)
    folder_parser.add_argument('-frames', help='Folder path for frames images')

    args = parser.parse_args()

    return args

def get_approach(name: str, method) -> ApproachBase:
    if name == APPROACH_LIST[0]:
        params = GDParameters(Method=Method[method])
        return GradientDescent(params)
    
    elif name == APPROACH_LIST[1]:
        params = KPParameters(Method=Method[method])
        return Keypoints(params)
    
    else:
        raise RuntimeError(f'{name} is incorrect approach name')
    
def init_logging_config(level: str, write_to_file: bool) -> None:
    log_level = level.upper()
    log_to_file = write_to_file

    log_format = '%(asctime)s | %(levelname)s | %(message)s'
    log_handlers = []

    if log_to_file:
        log_file_path = os.path.join(os.path.dirname(__file__), 'log.log')
        file_handler = logging.FileHandler(filename=log_file_path, mode='w')
        log_handlers.append(file_handler)
    
    else:
        log_handlers.append(logging.StreamHandler())

    logging.basicConfig(level=log_level, format=log_format, handlers=log_handlers)

def calc_scales(dataset: Dataset, approach: ApproachBase) -> ResultScales:

    is_valid, type = dataset.validate()

    if not is_valid:
        return ResultScales(
            ScaleX=0.0, ScaleY=0.0, SrcInfo=str(dataset), Message='not valid dataset'
        )

    is_image_flat = False

    if type == DatasetType.Frames:
        frame_1 = ExrUtils.read_image(dataset.Frame_1)
        is_image_flat |= ExrUtils.is_image_flat(frame_1)
        
        frame_2 = ExrUtils.read_image(dataset.Frame_2)
        is_image_flat |= ExrUtils.is_image_flat(frame_2)

        motion_vectors = ExrUtils.read_motion_vectors(dataset.MV_2)
        is_image_flat |= ExrUtils.is_image_flat(motion_vectors)

        evaluator = lambda: approach.from_frames(frame_1, frame_2, motion_vectors)

    else:
        motion_vectors_1 = ExrUtils.read_motion_vectors(dataset.MV_1)
        is_image_flat |= ExrUtils.is_image_flat(motion_vectors_1)

        motion_vectors_2 = ExrUtils.read_motion_vectors(dataset.MV_2)
        is_image_flat |= ExrUtils.is_image_flat(motion_vectors_2)

        evaluator = lambda: approach.from_motion_vectors(motion_vectors_1, motion_vectors_2)

    if is_image_flat:
        return ResultScales(
            ScaleX=0.0, ScaleY=0.0, SrcInfo=str(dataset), Message='one or more images are flat'
        )
    
    scale_x, scale_y = evaluator()

    return ResultScales(ScaleX=scale_x, ScaleY=scale_y, SrcInfo=str(dataset), Message='')

def is_directory_empty(path: str) -> bool:
    return path is None or not os.path.exists(path) or not os.path.isdir(path) or not bool(os.listdir(path))

def get_files_list(dir: str) -> list[str]:
    content = [os.path.join(dir, f) for f in os.listdir(dir)]
    return sorted(filter(lambda x: os.path.isfile(x), content))

def produce_datasets(args: argparse.Namespace) -> Iterator[Dataset]:
    if args.input_data == 'files':
        yield Dataset(args.frame_1, args.frame_2, args.mv_1, args.mv_2)
        return

    if is_directory_empty(args.mv):
        # raise ValueError('can not find any images with motion vectors')
        return
    
    mv_files = get_files_list(args.mv)
    frame_files = [''] * len(mv_files)

    if not is_directory_empty(args.frames):
        frame_files = get_files_list(args.frames)

    for idx in range(len(mv_files) - 1):
        yield Dataset(
            Frame_1=frame_files[idx],
            Frame_2=frame_files[idx + 1],
            MV_1=mv_files[idx],
            MV_2=mv_files[idx + 1]
        )

if __name__ == '__main__':

    args = parse_arguments()
    
    init_logging_config(args.log_level, args.log_to_file)
    logger = logging.getLogger(__name__)
    logger.info(f'Use module version: {module_version}')

    approach = get_approach(args.app, args.method)

    for dataset in produce_datasets(args):
        result = calc_scales(dataset, approach)
        logger.info(result)
