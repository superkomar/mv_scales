import argparse
import os

from utils import ExrUtils
from keypoints import KeypointsApproach
from gradient_descent import GradientDescentAlgorithm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-alg', help='Choose one of the following algorithms', choices=['keypoints', 'gradient descent'], required=True)
    parser.add_argument('-frame_1', help='File path for the first frame', required=True)
    parser.add_argument('-frame_2', help='File path for the second frame', required=True)
    parser.add_argument('-mv', help='File path for img with motion vectors', required=True)
    args = parser.parse_args()
    
    if args.alg == 'keypoints':
        algorithm = KeypointsApproach()
    elif args.alg == 'gradient descent':
        algorithm = GradientDescentAlgorithm()
    else:
        raise RuntimeError(f'Unknown algorithm: {args.alg}')

    algorithm.compute(
        frame_1=ExrUtils.read(args.frame_1),
        frame_2=ExrUtils.read(args.frame_2),
        mv = ExrUtils.red(args.mv)
    )
