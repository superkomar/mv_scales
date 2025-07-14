import os
import argparse

from mv_scales_compute import KeypointsAlgorithm
from mv_scales_compute import GradientDescentAlgorithm
from mv_scales_compute import utils

if __name__ == '__main__':

    DEFAULT_FRAME_PATH_1 = os.path.join(os.path.dirname(__file__), 'examples', 'frames', '02013.exr')
    DEFAULT_FRAME_PATH_2 = os.path.join(os.path.dirname(__file__), 'examples', 'frames', '02014.exr')
    DEFAULT_MV_PATH = os.path.join(os.path.dirname(__file__), 'examples', 'motion_vectors', '02014.exr')

    parser = argparse.ArgumentParser()
    parser.add_argument('-alg', help='Choose one of the following algorithms', choices=['keypoints', 'gradient descent'], required=True)
    parser.add_argument('-frame_1', help='File path for the first frame', default=DEFAULT_FRAME_PATH_1)
    parser.add_argument('-frame_2', help='File path for the second frame', default=DEFAULT_FRAME_PATH_2)
    parser.add_argument('-mv', help='File path for img with motion vectors', default=DEFAULT_MV_PATH)
    args = parser.parse_args()
    
    if args.alg == 'keypoints':
        algorithm = KeypointsAlgorithm()
    elif args.alg == 'gradient descent':
        algorithm = GradientDescentAlgorithm()
    else:
        raise RuntimeError(f'Unknown algorithm: {args.alg}')

    scale_x, scale_y = algorithm.compute_from_files(frame_1=args.frame_1, frame_2=args.frame_2, motion_vectors=args.mv)
    
    print(f'scale_x: {scale_x}')
    print(f'scale_y: {scale_y}')
