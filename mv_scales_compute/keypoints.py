import os
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, NamedTuple
import math

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from .utils import read_exr


class PixelCoords(NamedTuple):
    X: int
    Y: int
    
class MotionVector(NamedTuple):
    Coords: PixelCoords
    Vector: Tuple[float, float]

class KeypointsData(NamedTuple):
    Keypoints: Tuple[cv2.KeyPoint]
    Descriptors: npt.NDArray[np.float32]


class KeypointsAlgorithm():
    @staticmethod
    def compute_from_matrices(
        frame_1: npt.NDArray[np.float16], frame_2: npt.NDArray[np.float16], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        THRESHOLD = 500
        
        frame_keypoints_1 = KeypointsAlgorithm.detect_keypoints(frame_1)
        frame_keypoints_2 = KeypointsAlgorithm.detect_keypoints(frame_2)

        custom_motion_vectors = KeypointsAlgorithm.find_matches(frame_keypoints_1, frame_keypoints_2, THRESHOLD)
        scale_x, scale_y = KeypointsAlgorithm.compute_scales(custom_motion_vectors, motion_vectors)
        
        return scale_x, scale_y
    
    @staticmethod
    def compute_from_files(frame_1: str, frame_2: str, motion_vectors: str) -> Tuple[float, float]:
        return KeypointsAlgorithm.compute_from_matrices(
            frame_1=read_exr(frame_1),
            frame_2=read_exr(frame_2),
            motion_vectors=read_exr(motion_vectors)
        )
    
    @staticmethod
    def detect_keypoints(image: npt.NDArray[np.float16]) -> KeypointsData:
        grayscale = KeypointsAlgorithm.make_grayscale(image)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(grayscale, None)

        return KeypointsData(keypoints, descriptors)  

    @staticmethod
    def find_matches(img_1: KeypointsData, img_2: KeypointsData, threshold: int) -> List[MotionVector]:

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(img_1.Descriptors, img_2.Descriptors, k=2)

        factor = 0.0
        good_matches = []
        while len(good_matches) < threshold and factor < 0.9:
            factor += 0.05

            for m, n in matches:
                if m.distance > factor * n.distance:
                    continue
                
                pt_1 = PixelCoords(
                    X=int(img_1.Keypoints[m.queryIdx].pt[0]),
                    Y=int(img_1.Keypoints[m.queryIdx].pt[1])
                )
                
                pt_2 = PixelCoords(
                    X=int(img_2.Keypoints[m.trainIdx].pt[0]),
                    Y=int(img_2.Keypoints[m.trainIdx].pt[1])
                )

                if pt_1 == pt_2:
                    continue
                
                vector = (img_2.Keypoints[m.trainIdx].pt[0] - img_1.Keypoints[m.queryIdx].pt[0],
                          img_2.Keypoints[m.trainIdx].pt[1] - img_1.Keypoints[m.queryIdx].pt[1])
                
                good_matches.append(MotionVector(Coords=pt_2, Vector=vector))

        if len(good_matches) < threshold:
            raise RuntimeError('could not find enough keypoints on two images')
        
        print(f'last factor: {factor}')

        return good_matches

    @staticmethod
    def make_grayscale(img: npt.NDArray[np.float16]) -> npt.NDArray[np.uint8]:
        tone_mapping = cv2.createTonemapDrago(gamma=2.5, bias=0.85)
        tone_mapped_img = tone_mapping.process((img[:,:,:3]).astype(np.float32))
        tone_mapped_img = KeypointsAlgorithm._remove_nans(tone_mapped_img)
        tone_mapped_img = (tone_mapped_img * 255).astype(np.uint8)

        grayscale = cv2.cvtColor(tone_mapped_img, cv2.COLOR_RGB2GRAY)

        return grayscale

    @staticmethod
    def compute_scales(
        custom_motion_vectors: List[MotionVector], motion_vectors: npt.NDArray[np.float16], points_num: int = 10
    ) -> Tuple[float, float]:
        height = motion_vectors.shape[0]
        width = motion_vectors.shape[1]

        vectors = zip(
            [(v.Vector[1] / height, v.Vector[0] / width) for v in custom_motion_vectors],
            [motion_vectors[v.Coords.Y, v.Coords.X][:2] for v in custom_motion_vectors],
            # [pt for pt in matched_points] # for debug
        )

        vectors = sorted(vectors, reverse=True, key=lambda x: math.sqrt(x[1][0]**2 + x[1][1]**2))

        scale_x = 0.0
        scale_y = 0.0

        for vector in vectors[:points_num]:
            scale_y += vector[0][0] / (vector[1][0])
            scale_x += vector[0][1] / (vector[1][1])

        scale_x = scale_x / points_num
        scale_y = scale_y / points_num

        return scale_x, scale_y
    
    @staticmethod
    def _remove_nans(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        nan_checks = np.isnan(image)
        nan_pixels = np.argwhere(nan_checks)

        for pixel in nan_pixels:
            image[pixel[0], pixel[1], pixel[2]] = 0.0

        return image
