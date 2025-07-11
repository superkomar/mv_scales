import os
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import List, Tuple, NamedTuple
from collections import namedtuple
import math

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


class PixelCoords(NamedTuple):
    X: int
    Y: int

class MatchedPoints(NamedTuple):
    Start: PixelCoords
    End: PixelCoords

    def to_vector(self, max_y, max_x) -> Tuple[float, float]:
        return (
            (self.End.Y - self.Start.Y) / max_y,
            (self.End.X - self.Start.X) / max_x,
        )

class KeypointsData(NamedTuple):
    Keypoints: Tuple[cv2.KeyPoint]
    Descriptors: npt.NDArray[np.float32]
    GrayscaleImg: npt.NDArray[np.uint8]


class KeypointsApproach():
    @staticmethod
    def compute(
        frame_1: npt.NDArray[np.float16], frame_2: npt.NDArray[np.float16], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        frame_keypoints_1 = KeypointsApproach.detect_keypoints(frame_1)
        frame_keypoints_2 = KeypointsApproach.detect_keypoints(frame_2)

        matched_points = KeypointsApproach.find_matches(frame_keypoints_1, frame_keypoints_2, 500)

        scale_x, scale_y = KeypointsApproach.compute_scales(matched_points, motion_vectors)
        
        return scale_x, scale_y
    
    @staticmethod
    def detect_keypoints(image: npt.NDArray[np.float16]) -> KeypointsData:
        grayscale = KeypointsApproach.make_grayscale(image)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(grayscale, None)

        return KeypointsData(keypoints, descriptors, grayscale)  

    @staticmethod
    def find_matches(img_1: KeypointsData, img_2: KeypointsData, threshold: int) -> List[MatchedPoints]:

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

                good_matches.append(MatchedPoints(pt_1, pt_2))

        if len(good_matches) < threshold:
            raise RuntimeError('could not find enough keypoints on two images')
        
        print(f'factor: {factor}')

        return good_matches

    @staticmethod
    def make_grayscale(img: npt.NDArray[np.float16]) -> npt.NDArray[np.uint8]:
        tone_mapping = cv2.createTonemapDrago(gamma=2.5, bias=0.85)
        tone_mapped_img = tone_mapping.process((img[:,:,:3]).astype(np.float32))
        tone_mapped_img = KeypointsApproach._remove_nans(tone_mapped_img)
        tone_mapped_img = (tone_mapped_img * 255).astype(np.uint8)

        grayscale = cv2.cvtColor(tone_mapped_img, cv2.COLOR_RGB2GRAY)

        return grayscale

    @staticmethod
    def compute_scales(matched_points: List[MatchedPoints], motion_vectors: npt.NDArray[np.float16], points_num: int = 10) -> Tuple[float, float]:
        height = motion_vectors.shape[0]
        width = motion_vectors.shape[1]

        vectors = zip(
            [pt.to_vector(max_x=width, max_y=height) for pt in matched_points],
            [motion_vectors[pt.End.Y, pt.End.X][:2] for pt in matched_points],
            [pt for pt in matched_points] # for debug
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
        nan_pixels = np.argwhere(nan_checks == np.nan)

        for nan in nan_pixels:
            image[nan[0], nan[1], nan[2]] = 0.0

        return image
