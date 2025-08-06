import os
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, NamedTuple
import math
import logging

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from .utils import read_exr, write_exr, ImageUtils


class PixelCoords(NamedTuple):
    X: int
    Y: int

    @staticmethod
    def to_vec(start: "PixelCoords", end: "PixelCoords") -> Tuple[int, int]:
        return (end.X - start.X, end.Y - start.Y)
    
class MotionVector(NamedTuple):
    Coords: PixelCoords
    Vector: Tuple[float, float]

class KeypointsData(NamedTuple):
    Keypoints: Tuple[cv2.KeyPoint]
    Descriptors: npt.NDArray[np.float32]
    Source: npt.NDArray[np.float16] # for debug

class MatchedPoints(NamedTuple):
    Relation: float
    Point_1: PixelCoords
    Point_2: PixelCoords


class KeypointsApproach():
    _MATCHES_THRESHOLD_ = 10
    _NUM_POINTS_TO_COMPUTE_ = 5

    def __init__(self, matches_threshold: int = _MATCHES_THRESHOLD_, num_points_to_compare: int = _NUM_POINTS_TO_COMPUTE_) -> None:
        self._matches_threshold = matches_threshold
        self._num_points_to_compare = num_points_to_compare

        self.logger = logging.getLogger(__name__)

    def compute_from_motion_vectors(self, mv_1: npt.NDArray[np.float16], mv_2: npt.NDArray[np.float16]) -> Tuple[float, float]:
        self.logger.info('Keypoints (motion vectors) has started')

        mv_1 = mv_1[..., :2]
        mv_2 = mv_2[..., :2]

        mv_8bit_1 = self.make_8bit(mv_1)
        mv_8bit_2 = self.make_8bit(mv_2)
        
        mv_keypoints_1 = self.detect_keypoints(mv_8bit_1)
        mv_keypoints_2 = self.detect_keypoints(mv_8bit_2)

        # custom_motion_vectors = KeypointsAlgorithm.find_matches(mv_keypoints_1, mv_keypoints_2, KeypointsAlgorithm.MATCHES_THRESHOLD)
        custom_motion_vectors = self._find_matches_new(mv_keypoints_1, mv_keypoints_2)

        scale_x, scale_y = self._compute_scales(custom_motion_vectors, mv_2)

        self.logger.info('Keypoints (motion vectors) has ended')

        return scale_x, scale_y    

    def compute_from_frames(
        self, frame_1: npt.NDArray[np.float16], frame_2: npt.NDArray[np.float16], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
        
        self.logger.info('Keypoints (frames) has started')
        
        grayscale_frame_1 = self.make_grayscale(frame_1)
        grayscale_frame_2 = self.make_grayscale(frame_2)

        frame_keypoints_1 = self.detect_keypoints(grayscale_frame_1)
        frame_keypoints_2 = self.detect_keypoints(grayscale_frame_2)

        custom_motion_vectors = self._find_matches_new(frame_keypoints_1, frame_keypoints_2)
        scale_x, scale_y = self._compute_scales(custom_motion_vectors, motion_vectors)

        self.logger.info('Keypoints (frames) has ended')
        
        return scale_x, scale_y
    
    @staticmethod
    def detect_keypoints(image_8bit: npt.NDArray[np.float16]) -> KeypointsData:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image_8bit, None)

        return KeypointsData(Keypoints=keypoints, Descriptors=descriptors, Source=image_8bit)
    
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
                MotionVector(Coords=pt_2, Vector=PixelCoords.to_vec(start=pt_1, end=pt_2)),
                m # for debug
            )

        matched_points = sorted(list(matched_points.values()), key=lambda x: x[0])

        if self.logger.getEffectiveLevel() == 10:
            good_keypoints = [[x[2]] for x in matched_points]
            matches_img = cv2.drawMatchesKnn(
                kpd_1.Source, kpd_1.Keypoints,
                kpd_2.Source, kpd_2.Keypoints,
                good_keypoints[:5],
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite('debug\\matches_new.jpg', matches_img)

        if len(matched_points) < self._matches_threshold:
            print(f'{len(matched_points)=}')
            raise RuntimeError(f'find only {len(matched_points)}, but needed {self._matches_threshold}')
        
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

        if len(good_matches) < self._matches_threshold:
            msg = f'Have found only {len(good_matches)} matches, but minimum needed are {self._matches_threshold}'
            self.logger.error(msg)
            raise RuntimeError(msg)
        
        self.logger.info(f'last factor: {factor}')

        matches_img = cv2.drawMatchesKnn(
            kpd_1.Source, kpd_1.Keypoints,
            kpd_2.Source, kpd_2.Keypoints,
            good_keypoints[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        if self.logger.getEffectiveLevel() == 10:
            cv2.imwrite('debug\\matches.jpg', matches_img)

        return good_matches

    @staticmethod
    def make_grayscale(img: npt.NDArray[np.float16]) -> npt.NDArray[np.uint8]:
        tone_mapping = cv2.createTonemapDrago(gamma=2.5, bias=0.85)
        tone_mapped_img = tone_mapping.process((img[:,:,:3]).astype(np.float32))
        tone_mapped_img = ImageUtils.remove_nans(tone_mapped_img)
        tone_mapped_img = (tone_mapped_img * 255).astype(np.uint8)

        grayscale = cv2.cvtColor(tone_mapped_img, cv2.COLOR_RGB2GRAY)

        return grayscale
    
    @staticmethod
    def make_8bit(mv: npt.NDArray[np.float16]) -> npt.NDArray[np.float16]:
        min = mv.min()
        max = mv.max()

        normalized_mv = (mv - min) / (max - min)
        normalized_mv = (normalized_mv * 255.0).astype(np.uint8)

        mv_8bit = normalized_mv[..., 0:1] * 0.5 + normalized_mv[..., 1:2] * 0.5
        
        return mv_8bit.astype(np.uint8)

    
    def _compute_scales(
        self, custom_motion_vectors: List[MotionVector], motion_vectors: npt.NDArray[np.float16]
    ) -> Tuple[float, float]:
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

        for vector in custom_motion_vectors[:self._num_points_to_compare]:
            coords = vector.Coords
            custom_vec = vector.Vector

            # mv = motion_vectors[]

        for vector in vectors[:self._num_points_to_compare]:
            print(f'coords | x: {vector[2].X};  y: {vector[2].Y}')
            print(f'old_mv | x: {vector[0][1]}; y: {vector[0][0]}')
            print(f'new_mv | x: {vector[1][1] / height}; y: {vector[1][0] / width}')
            print()

            scale_y += vector[1][0] / height / vector[0][0]
            scale_x += vector[1][1] / width  / vector[0][1]

        scale_x = scale_x / self._num_points_to_compare
        scale_y = scale_y / self._num_points_to_compare

        return scale_x, scale_y
