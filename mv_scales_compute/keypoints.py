import os
import numpy as np
import numpy.typing as npt
from typing import Tuple, NamedTuple, Dict
from enum import Enum
from dataclasses import dataclass

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from .utils import ExrUtils, ImageUtils
from .approach_base import ApproachBase, ApproachParameters, Method


class PixelCoords(NamedTuple):
    X: int
    Y: int

    def to(self, end: "PixelCoords") -> "PixelCoords":
        return PixelCoords(X=(end.X - self.X), Y=(end.Y - self.Y))


class MotionVector(NamedTuple):
    Distance: float
    Coord: PixelCoords
    Vector: PixelCoords
    Match: cv2.DMatch # for debug


class Detector(Enum):
    SIFT = 1
    AKAZE = 2


@dataclass
class KPParameters(ApproachParameters):
    MatchesThreshold: int = 50
    IsMovingBackward: bool = True
    
    FactorForFrames: float = 0.7
    FactorForMv: float = 0.5


class Keypoints(ApproachBase):

    def __init__(self, parameters: KPParameters = KPParameters()) -> None:
        super().__init__(parameters)

        self._matches_threshold = parameters.MatchesThreshold
        self._is_moving_backward = parameters.IsMovingBackward
        self._zero_eps = parameters.ZeroEpsilon
        
        self._factor_frames = parameters.FactorForFrames
        self._factor_mv = parameters.FactorForMv

    def from_motion_vectors(
            self, motion_vectors_1: npt.NDArray[np.float32], motion_vectors_2: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:

        motion_vectors_1 = motion_vectors_1[..., :2]
        motion_vectors_2 = motion_vectors_2[..., :2]

        mv_8bit_1 = ImageUtils.make_8bit(motion_vectors_1)
        mv_8bit_2 = ImageUtils.make_8bit(motion_vectors_2)

        scale_x, scale_y = self._compute_scales(
            source_8bit=mv_8bit_1,
            target_8bit=mv_8bit_2,
            original_mv=motion_vectors_2,
            factor=self._factor_mv
        )

        return scale_x, scale_y

    def from_frames(
            self,
            frame_1: npt.NDArray[np.float32],
            frame_2: npt.NDArray[np.float32],
            motion_vectors: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:

        grayscale_1 = ImageUtils.make_grayscale(frame_1)
        grayscale_2 = ImageUtils.make_grayscale(frame_2)

        motion_vectors = motion_vectors[..., :2]

        scale_x, scale_y = self._compute_scales(
            source_8bit=grayscale_2,
            target_8bit=grayscale_1,
            original_mv=motion_vectors,
            factor=self._factor_frames
        )

        return scale_x, scale_y

    def _compute_scales(
            self,
            source_8bit: npt.NDArray[np.uint8],
            target_8bit: npt.NDArray[np.uint8],
            original_mv: npt.NDArray[np.float32],
            factor: float
    ) -> Tuple[float, float]:

        custom_mv = {}

        # self._find_matches(source_8bit, target_8bit, custom_mv, Detector.SIFT, factor)
        # self._find_matches_test(source_8bit, target_8bit, custom_mv, Detector.SIFT, factor)

        for detector in Detector:
            self._find_matches(source_8bit, target_8bit, custom_mv, detector, factor)

        #####################
        # source_frame = source_8bit.copy().astype(np.float32)
        # target_frame = target_8bit.copy().astype(np.float32)

        # src_coords = np.array([pt for pt in custom_mv.keys()])
        # source_frame[src_coords[..., 1], src_coords[..., 0]] = -100
        # ExrUtils.write_exr(source_frame, 'debug\\src_frame.exr')

        # trg_coords = np.array([((k.X + v.X), (k.Y + v.Y)) for k, v in custom_mv.items()])
        # target_frame[trg_coords[..., 1], trg_coords[..., 0]] = -100
        # ExrUtils.write_exr(target_frame, 'debug\\trg_frame.exr')
        ################

        scale_x, scale_y = self._calc_motion_vectors_scales(custom_mv, original_mv)

        return scale_x, scale_y

    def _find_matches_test(
            self,
            source: npt.NDArray[np.uint8],
            target: npt.NDArray[np.uint8],
            custom_mv: Dict[PixelCoords, PixelCoords],
            detector_type: Detector,
            factor: float
    ) -> None:
        
        detector = self._get_detector(detector_type)

        source_keypoints, source_descriptors = detector.detectAndCompute(source, None)
        target_keypoints, target_descriptors = detector.detectAndCompute(target, None)
        
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches_from_src = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
        matches_from_trg = matcher.knnMatch(target_descriptors, source_descriptors, k=2)

        knn_ratio = lambda matches: [m for m, n in matches if m.distance < factor * n.distance]

        good_src_trg = knn_ratio(matches_from_src)
        good_trg_src = knn_ratio(matches_from_trg)

        rev = {m.trainIdx: m.queryIdx for m in good_trg_src}

        mutual = [m for m in good_src_trg if rev.get(m.trainIdx, -1) == m.queryIdx]

        # matches = sorted(matches, key=lambda x: x.distance)

        motion_vectors = {}
        for m in mutual:

            source_pt = self._pixel_coords_from_kpd(source_keypoints[m.queryIdx])
            target_pt = self._pixel_coords_from_kpd(target_keypoints[m.trainIdx])

            if source_pt == target_pt or target_pt in custom_mv:
                continue

            mv = MotionVector(
                Distance=m.distance,
                Coord=source_pt,
                Vector=source_pt.to(target_pt),
                Match=m,
            )

            if mv not in motion_vectors:
                motion_vectors[mv.Coord] = mv
            
            elif mv.Match.distance < motion_vectors[mv.Coord].Match.distance:
                motion_vectors[mv.Coord] = mv

        motion_vectors = sorted(list(motion_vectors.values()), key=lambda x: x.Distance)
        self.logger.debug(f'Keypoints matched: {len(motion_vectors)}; detector: {detector_type.name}')

        good_matches_count = int(len(motion_vectors) / 2)
        good_matches_count = self._matches_threshold 
        custom_mv.update({el.Coord: el.Vector for el in motion_vectors[:good_matches_count]})

        if self._is_debug:
            good_keypoints = [[x.Match] for x in motion_vectors[:good_matches_count]]
            matches_img = cv2.drawMatchesKnn(
                source, source_keypoints,
                target, target_keypoints,
                good_keypoints,
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(f'debug\\matches_{detector_type.name}.jpg', matches_img)
            
            img = cv2.drawKeypoints(source, source_keypoints, source, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(f'debug\\{detector_type.name}_keypoints.jpg', img)


    def _find_matches(
            self,
            source: npt.NDArray[np.uint8],
            target: npt.NDArray[np.uint8],
            custom_mv: Dict[PixelCoords, PixelCoords],
            detector_type: Detector,
            factor: float
    ) -> None:
        
        detector = self._get_detector(detector_type)

        source_keypoints, source_descriptors = detector.detectAndCompute(source, None)
        target_keypoints, target_descriptors = detector.detectAndCompute(target, None)
        
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(source_descriptors, target_descriptors, k=2)

        motion_vectors = {}
        for m, n in matches:

            if m.distance >= factor * n.distance:
                continue

            source_pt = self._pixel_coords_from_kpd(source_keypoints[m.queryIdx])
            target_pt = self._pixel_coords_from_kpd(target_keypoints[m.trainIdx])

            if source_pt == target_pt or target_pt in custom_mv:
                continue

            mv = MotionVector(
                Distance=(m.distance / n.distance),
                Coord=source_pt,
                Vector=source_pt.to(target_pt),
                Match=m,
            )

            if mv not in motion_vectors:
                motion_vectors[mv.Coord] = mv
            
            elif mv.Match.distance < motion_vectors[mv.Coord].Match.distance:
                motion_vectors[mv.Coord] = mv

        motion_vectors = sorted(list(motion_vectors.values()), key=lambda x: x.Distance)
        self.logger.debug(f'Keypoints matched: {len(motion_vectors)}; detector: {detector_type.name}')

        good_matches_count = int(len(motion_vectors) / 2)
        good_matches_count = self._matches_threshold 
        custom_mv.update({el.Coord: el.Vector for el in motion_vectors[:good_matches_count]})

        if self._is_debug:
            good_keypoints = [[x.Match] for x in motion_vectors[:good_matches_count]]
            matches_img = cv2.drawMatchesKnn(
                source, source_keypoints,
                target, target_keypoints,
                good_keypoints,
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(f'debug\\matches_{detector_type.name}.jpg', matches_img)
            
            img = cv2.drawKeypoints(source, source_keypoints, source, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(f'debug\\{detector_type.name}_keypoints.jpg', img)

    
    def _calc_motion_vectors_scales(
        self, custom_mv: Dict[PixelCoords, PixelCoords], original_mv: npt.NDArray[np.float32]
    ) -> Tuple[float, float]:

        height, width = original_mv.shape[:2]

        coords = np.array([point for point in custom_mv.keys()])
        original_mv = original_mv[coords[..., 1], coords[..., 0]]

        custom_mv = np.array([(mv.X / width, mv.Y / height) for mv in custom_mv.values()])

        self.logger.debug(f'Vectors to compare: {len(coords)}')

        # not_zero_mask_x = (abs(original_mv[..., 0]) > self._zero_eps) & (abs(custom_mv[..., 0]) > self._zero_eps)
        # not_zero_mask_y = (abs(original_mv[..., 1]) > self._zero_eps) & (abs(custom_mv[..., 1]) > self._zero_eps)
            
        # for idx in range(coords.shape[0]):

        #     if not not_zero_mask_x[idx] and not not_zero_mask_y[idx]:
        #         continue

        #     if not_zero_mask_x[idx] and not_zero_mask_y[idx]:
        #         self.logger.debug(f'coord    | x={coords[idx, 0]}; y={coords[idx, 1]}')
        #         self.logger.debug(f'custom   | x={custom_mv[idx, 0]:0.6f}; y={custom_mv[idx, 1]:0.6f}')
        #         self.logger.debug(f'original | x={original_mv[idx, 0]:0.6f}; y={original_mv[idx, 1]:0.6f}')
            
        #     elif not_zero_mask_x[idx]:
        #         self.logger.debug(f'coord    | x={coords[idx, 0]}; y={coords[idx, 1]}')
        #         self.logger.debug(f'custom   | x={custom_mv[idx, 0]:0.6f}')
        #         self.logger.debug(f'original | x={original_mv[idx, 0]:0.6f}')

        #     else:
        #         self.logger.debug(f'coord    | x={coords[idx, 0]}; y={coords[idx, 1]}')
        #         self.logger.debug(f'custom   | y={custom_mv[idx, 1]:0.6f}')
        #         self.logger.debug(f'original | y={original_mv[idx, 1]:0.6f}')


        #     self.logger.debug('')

        return self.calculate_scales(custom_mv, original_mv)
    
    @staticmethod
    def _get_detector(detector):
        if detector == Detector.AKAZE: return cv2.AKAZE_create()
        else: return cv2.SIFT_create()
    
    @staticmethod
    def _pixel_coords_from_kpd(kpd: cv2.KeyPoint) -> PixelCoords:
        pt_x, pt_y = ImageUtils.bilinear_interpolation(x=kpd.pt[0], y=kpd.pt[1])
        # pt_x, pt_y = int(kpd.pt[0]), int(kpd.pt[1])
        return PixelCoords(X=pt_x, Y=pt_y)
