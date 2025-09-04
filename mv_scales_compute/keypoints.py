import os
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, NamedTuple
import math
import logging
from dataclasses import dataclass

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from .utils import ExrUtils, ImageUtils
from .approach_base import ApproachBase


class PixelCoords(NamedTuple):
    X: int
    Y: int

    def __sub__(self, other: "PixelCoords") -> "PixelCoords":
        return PixelCoords(self.X - other.X, self.Y - other.Y)

    
class MotionVector(NamedTuple):
    Coords: PixelCoords
    Vector: PixelCoords


class KeypointsData(NamedTuple):
    Keypoints: Tuple[cv2.KeyPoint]
    Descriptors: npt.NDArray[np.float32]
    Source: npt.NDArray[np.uint8] # for debug


@dataclass
class ApproachParameters:
    MatchesThreshold: int = 50
    IsMovingBackward: bool = True
    ScaleZeroEpsilon: float = 1e-6
    DetectAlgorithm: str = 'sift'


class KeypointsApproach(ApproachBase):

    def __init__(self, parameters: ApproachParameters = ApproachParameters()) -> None:
        super().__init__()

        self._matches_threshold = parameters.MatchesThreshold
        self._is_moving_backward = parameters.IsMovingBackward
        self._zero_eps = parameters.ScaleZeroEpsilon
        self._detect_alg = parameters.DetectAlgorithm
    
    def from_motion_vectors(
        self, motion_vectors_1: npt.NDArray[np.float16], motion_vectors_2: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        self.logger.info('Keypoints (motion vectors) has started')

        motion_vectors_1 = motion_vectors_1[..., :2]
        motion_vectors_2 = motion_vectors_2[..., :2]

        mv_8bit_1 = ImageUtils.make_8bit(motion_vectors_1)
        mv_8bit_2 = ImageUtils.make_8bit(motion_vectors_2)

        scale_x, scale_y = self._compute_scales(
            source_8bit=mv_8bit_1,
            target_8bit=mv_8bit_2,
            motion_vectors=motion_vectors_2
        )

        self.logger.info('Keypoints (motion vectors) has ended')

        return scale_x, scale_y    

    def from_frames(
        self, frame_1: npt.NDArray[np.float16], frame_2: npt.NDArray[np.float16], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        self.logger.info('Keypoints (frames) has started')
        
        grayscale_1 = ImageUtils.make_grayscale(frame_1)
        grayscale_2 = ImageUtils.make_grayscale(frame_2)

        motion_vectors = motion_vectors[..., :2]

        scale_x, scale_y = self._compute_scales(
            source_8bit=grayscale_1,
            target_8bit=grayscale_2,
            motion_vectors=motion_vectors
        )

        self.logger.info('Keypoints (frames) has ended')
        
        return scale_x, scale_y

    def _compute_scales(
            self, source_8bit: npt.NDArray[np.uint8], target_8bit: npt.NDArray[np.uint8], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:

        source_keypoints = self._detect_keypoints(source_8bit, self._detect_alg)
        target_keypoints = self._detect_keypoints(target_8bit, self._detect_alg)

        custom_motion_vectors = self._find_matches(source_keypoints, target_keypoints)

        scale_x, scale_y = self._calc_motion_vectors_scales(custom_motion_vectors, motion_vectors)

        return scale_x, scale_y

    def _find_matches(self, source: KeypointsData, target: KeypointsData) -> List[MotionVector]:
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(source.Descriptors, target.Descriptors, k=2)

        factor = 0.7
        matched_points = {}
        for m, n in matches:

            if m.distance >= factor * n.distance:
                continue

            source_pt = self._pixel_coords_from_kpd(source.Keypoints[m.queryIdx])
            target_pt = self._pixel_coords_from_kpd(target.Keypoints[m.trainIdx])

            if source_pt == target_pt or target_pt in matched_points:
                continue

            matched_points[target_pt] = (
                m.distance / n.distance,
                MotionVector(Coords=target_pt, Vector=(target_pt - source_pt)),
                m # debug purposes
            )

        matched_points = sorted(list(matched_points.values()), key=lambda x: x[0])
        self.logger.debug(f'Have found {len(matched_points)} keypoints matches')

        if self._is_debug:
            good_keypoints = [[x[2]] for x in matched_points]
            matches_img = cv2.drawMatchesKnn(
                source.Source, source.Keypoints,
                target.Source, target.Keypoints,
                good_keypoints[:self._matches_threshold],
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite('debug\\matches_new.jpg', matches_img)

        if len(matched_points) < self._matches_threshold:
            message = f'matched only {len(matched_points)} keypoints, but needed {self._matches_threshold}'
            self.logger.error(message)
            raise RuntimeError(message)
        
        return [match[1] for match in matched_points[:self._matches_threshold]]
    
    def _calc_motion_vectors_scales(
        self, custom_mv: List[MotionVector], original_mv: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:

        height, width = original_mv.shape[:2]

        coords = np.array([mv.Coords for mv in custom_mv])
        original_mv = original_mv[coords[..., 1], coords[..., 0]]

        custom_mv = np.array([(mv.Vector.Y / height, mv.Vector.X / width) for mv in custom_mv])

        return self.calculate_mv_scales(custom_mv, original_mv, self._zero_eps)
    
    @staticmethod
    def _detect_keypoints(image_8bit: npt.NDArray[np.uint8], alg: str = 'sift') -> KeypointsData:
        if alg.lower() == 'akaze':   detector = cv2.AKAZE_create()  
        elif alg.lower() == 'brisk': detector = cv2.BRISK_create()
        elif alg.lower() == 'sift':  detector = cv2.SIFT_create()
        else: raise RuntimeError(f'{alg} is not supported')

        keypoints, descriptors = detector.detectAndCompute(image_8bit, None)

        return KeypointsData(Keypoints=keypoints, Descriptors=descriptors, Source=image_8bit)
    
    @staticmethod
    def _pixel_coords_from_kpd(kpd: cv2.KeyPoint) -> PixelCoords:
        source_pt_x, source_pt_y = ImageUtils.bilinear_interpolation(x=kpd.pt[0], y=kpd.pt[1])
        return PixelCoords(source_pt_x, source_pt_y)
