import os
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, NamedTuple
import math
import logging
from dataclasses import dataclass

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from .utils import read_exr, write_exr, ImageUtils


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


class MatchedPoints(NamedTuple):
    Relation: float
    Point_1: PixelCoords
    Point_2: PixelCoords


@dataclass
class ApproachParameters:
    MatchesThreshold = 50
    NumPointsToCompare = 10
    IsMovingBackward = True


class KeypointsApproach:
    _MATCHES_THRESHOLD_ = 50
    _NUM_POINTS_TO_COMPUTE_ = 10

    def __init__(self, parameters: ApproachParameters = ApproachParameters()) -> None:
        self._matches_threshold = parameters.MatchesThreshold
        self._num_points_to_compare = parameters.NumPointsToCompare
        self._is_moving_backward = parameters.IsMovingBackward

        self.logger = logging.getLogger(__name__)
        self._is_debug = self.logger.getEffectiveLevel() == 10
    
    def from_motion_vectors(
        self, mv_1: npt.NDArray[np.float16], mv_2: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        self.logger.info('Keypoints (motion vectors) has started')

        mv_8bit_1 = ImageUtils.make_8bit(mv_1)
        mv_8bit_2 = ImageUtils.make_8bit(mv_2)

        motion_vectors = mv_2 if self._is_moving_backward else mv_1

        scale_x, scale_y = self._compute_scales(mv_8bit_1, mv_8bit_2, motion_vectors)

        self.logger.info('Keypoints (motion vectors) has ended')

        return scale_x, scale_y    

    def from_frames(
        self, frame_1: npt.NDArray[np.float16], frame_2: npt.NDArray[np.float16], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        self.logger.info('Keypoints (frames) has started')
        
        grayscale_1 = ImageUtils.make_grayscale(frame_1)
        grayscale_2 = ImageUtils.make_grayscale(frame_2)

        scale_x, scale_y = self._compute_scales(grayscale_1, grayscale_2, motion_vectors)

        self.logger.info('Keypoints (frames) has ended')
        
        return scale_x, scale_y

    def _compute_scales(
            self, img_8bit_1: npt.NDArray[np.uint8], img_8bit_2: npt.NDArray[np.uint8], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:

        self.logger.info(f'Keypoints has started')

        frame_keypoints_1 = self.detect_keypoints(img_8bit_1)
        frame_keypoints_2 = self.detect_keypoints(img_8bit_2)

        custom_motion_vectors = self._find_matches_new(frame_keypoints_1, frame_keypoints_2)
        self.find_matches(frame_keypoints_1, frame_keypoints_2)

        scale_x, scale_y = self._calc_motion_vectors_scales_new(custom_motion_vectors, motion_vectors)
        self.logger.debug(f'{scale_x=}; {scale_y=}')

        # scale_x, scale_y = self._calc_motion_vectors_scales(custom_motion_vectors, motion_vectors)
        # self.logger.debug(f'{scale_x=}; {scale_y=}')

        self.logger.info('Keypoints has completed')

        return scale_x, scale_y
    
    def _find_matches_new(self, kpd_1: KeypointsData, kpd_2: KeypointsData) -> List[MotionVector]:
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(kpd_1.Descriptors, kpd_2.Descriptors, k=2)

        matched_points = {}
        for m, n in matches:
            pt_1_x, pt_1_y = ImageUtils.bilinear_interpolation(
                x=kpd_1.Keypoints[m.queryIdx].pt[0],
                y=kpd_1.Keypoints[m.queryIdx].pt[1]
            )
            pt_1 = PixelCoords(pt_1_x, pt_1_y)

            pt_2_x, pt_2_y = ImageUtils.bilinear_interpolation(
                x=kpd_2.Keypoints[m.trainIdx].pt[0],
                y=kpd_2.Keypoints[m.trainIdx].pt[1]
            )
            pt_2 = PixelCoords(pt_2_x, pt_2_y)

            if pt_1 == pt_2 or pt_2 in matched_points:
                continue

            matched_points[pt_2] = (
                m.distance / n.distance,
                MotionVector(Coords=pt_2, Vector=(pt_2 - pt_1)),
                m # for debug
            )

        matched_points = sorted(list(matched_points.values()), key=lambda x: x[0])
        self.logger.debug(f'new alg found {len(matched_points)=} keypoints')

        if self._is_debug:
            good_keypoints = [[x[2]] for x in matched_points]
            matches_img = cv2.drawMatchesKnn(
                kpd_1.Source, kpd_1.Keypoints,
                kpd_2.Source, kpd_2.Keypoints,
                good_keypoints[:self._num_points_to_compare],
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite('debug\\matches_new.jpg', matches_img)

        if len(matched_points) < self._matches_threshold:
            message = f'find only {len(matched_points)}, but needed {self._matches_threshold}'
            self.logger.error(message)
            raise RuntimeError(message)
        
        return [match[1] for match in matched_points]

    def find_matches(self, kpd_1: KeypointsData, kpd_2: KeypointsData) -> List[MotionVector]:

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(kpd_1.Descriptors, kpd_2.Descriptors, k=2)

        factor = 0.0
        good_matches = []
        good_keypoints = []
        while len(good_matches) < self._matches_threshold and factor < 0.9:
            factor += 0.05
            
            good_keypoints.clear()

            for m, n in matches:
                if m.distance > factor * n.distance:
                    continue
                
                # pt_1_x = kpd_1.Keypoints[m.queryIdx].pt[0]
                # pt_1_y = kpd_1.Keypoints[m.queryIdx].pt[1]
                pt_1_x, pt_1_y = ImageUtils.bilinear_interpolation(
                    x=kpd_1.Keypoints[m.queryIdx].pt[0],
                    y=kpd_1.Keypoints[m.queryIdx].pt[1]
                )
                pt_1 = PixelCoords(pt_1_x, pt_1_y)

                # pt_2_x = kpd_2.Keypoints[m.trainIdx].pt[0]
                # pt_2_y = kpd_2.Keypoints[m.trainIdx].pt[1]
                pt_2_x, pt_2_y = ImageUtils.bilinear_interpolation(
                    x=kpd_2.Keypoints[m.trainIdx].pt[0],
                    y=kpd_2.Keypoints[m.trainIdx].pt[1]
                )
                pt_2 = PixelCoords(pt_2_x, pt_2_y)

                if pt_1 == pt_2:
                    continue

                vector = (pt_2_x - pt_1_x, pt_2_y - pt_1_y)
                
                good_matches.append(MotionVector(Coords=pt_2, Vector=vector))
                good_keypoints.append([m])

        self.logger.debug(f'old alg found {len(good_keypoints)=} keypoints')

        if self._is_debug:
            matches_img = cv2.drawMatchesKnn(
                kpd_1.Source, kpd_1.Keypoints,
                kpd_2.Source, kpd_2.Keypoints,
                good_keypoints[:self._num_points_to_compare],
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite('debug\\matches_old.jpg', matches_img)

        if len(good_matches) < self._matches_threshold:
            msg = f'Have found only {len(good_matches)} matches, but minimum needed are {self._matches_threshold}'
            self.logger.error(msg)
            raise RuntimeError(msg)
        
        self.logger.info(f'last factor: {factor}')

        return good_matches
    
    def _calc_motion_vectors_scales(
        self, custom_motion_vectors: List[MotionVector], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        motion_vectors = motion_vectors[..., :2]

        height = motion_vectors.shape[0]
        width = motion_vectors.shape[1]

        vectors = zip(
            [motion_vectors[v.Coords.Y, v.Coords.X] for v in custom_motion_vectors],
            [(v.Vector[1], v.Vector[0]) for v in custom_motion_vectors],
            [pt.Coords for pt in custom_motion_vectors] # for debug
        )

        vectors = sorted(vectors, reverse=True, key=lambda x: math.sqrt(x[0][0]**2 + x[0][1]**2))

        scale_x = 0.0
        scale_y = 0.0

        for vector in vectors[:self._num_points_to_compare]:
            # print(f'coords | x: {vector[2].X};  y: {vector[2].Y}')
            # print(f'old_mv | x: {vector[0][1]}; y: {vector[0][0]}')
            # print(f'new_mv | x: {vector[1][1] / height}; y: {vector[1][0] / width}')
            # print()

            scale_y += vector[1][0] / height / vector[0][0]
            scale_x += vector[1][1] / width  / vector[0][1]

        scale_x /= self._num_points_to_compare
        scale_y /= self._num_points_to_compare

        return scale_x, scale_y
    
    def _calc_motion_vectors_scales_new(
        self, custom_motion_vectors: List[MotionVector], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        motion_vectors = motion_vectors[..., :2]

        height = motion_vectors.shape[0]
        width = motion_vectors.shape[1]

        scale_x = 0.0
        scale_y = 0.0

        eps = 1e-10

        num_vectors_to_compare = 0
        for vector in custom_motion_vectors:
            origin_mv_y, origin_mv_x = motion_vectors[vector.Coords.Y, vector.Coords.X]

            if abs(origin_mv_x) < eps or abs(origin_mv_y) < eps:
                continue

            scale_x += float(vector.Vector.X) / origin_mv_x / width
            scale_y += float(vector.Vector.Y) / origin_mv_y / height

            num_vectors_to_compare += 1

            if num_vectors_to_compare >= self._num_points_to_compare:
                break

            self.logger.debug(f'coords | x: {vector.Coords.X}; y: {vector.Coords.Y}')
            self.logger.debug(f'old_mv | x: {origin_mv_x:.8f}; y: {origin_mv_y:.8f}')
            self.logger.debug(f'new_mv | x: {vector.Vector.X / width:.8f}; y: {vector.Vector.Y / height:.8f}')
            self.logger.debug('')

        scale_x /= num_vectors_to_compare
        scale_y /= num_vectors_to_compare

        return scale_x, scale_y
    
    @staticmethod
    def detect_keypoints(image_8bit: npt.NDArray[np.uint8]) -> KeypointsData:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image_8bit, None)

        return KeypointsData(Keypoints=keypoints, Descriptors=descriptors, Source=image_8bit)
