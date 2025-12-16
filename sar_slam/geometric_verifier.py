"""
Geometric Motion Verification Module

Verifies actual motion using geometric methods and 3D geometry constraints.
This module provides geometric evidence for motion detection.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class MotionHypothesis:
    """Represents a motion hypothesis with geometric evidence."""
    is_moving: bool
    confidence: float
    epipolar_error: float
    feature_count: int
    geometric_consistency: float


class GeometricMotionVerifier:
    """
    Verifies motion using geometric constraints and 3D structure.
    
    This verifier uses:
    - Epipolar geometry
    - Fundamental matrix estimation
    - Motion consistency checks
    - 3D reconstruction verification
    """
    
    def __init__(
        self,
        ransac_threshold: float = 1.0,
        min_inliers: int = 8,
        confidence: float = 0.99,
        motion_threshold: float = 2.0
    ):
        """
        Initialize the geometric motion verifier.
        
        Args:
            ransac_threshold: RANSAC reprojection threshold
            min_inliers: Minimum number of inliers for valid geometry
            confidence: RANSAC confidence level
            motion_threshold: Threshold for motion magnitude
        """
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.confidence = confidence
        self.motion_threshold = motion_threshold
        
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Initialize feature detector and descriptor
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def verify(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        motion_mask: Optional[np.ndarray] = None
    ) -> MotionHypothesis:
        """
        Verify motion using geometric constraints.
        
        Args:
            frame1: Previous frame
            frame2: Current frame
            motion_mask: Optional mask from motion detection
            
        Returns:
            MotionHypothesis with verification results
        """
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1.copy()
            gray2 = frame2.copy()
        
        # Detect and match features
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < self.min_inliers:
            return MotionHypothesis(
                is_moving=False,
                confidence=0.0,
                epipolar_error=float('inf'),
                feature_count=0,
                geometric_consistency=0.0
            )
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_inliers:
            return MotionHypothesis(
                is_moving=False,
                confidence=0.0,
                epipolar_error=float('inf'),
                feature_count=len(good_matches),
                geometric_consistency=0.0
            )
        
        # Extract matched point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Compute fundamental matrix using RANSAC
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            cv2.FM_RANSAC,
            self.ransac_threshold,
            self.confidence
        )
        
        if F is None or mask is None:
            return MotionHypothesis(
                is_moving=False,
                confidence=0.0,
                epipolar_error=float('inf'),
                feature_count=len(good_matches),
                geometric_consistency=0.0
            )
        
        # Analyze geometric consistency
        inliers = mask.sum()
        inlier_ratio = inliers / len(good_matches)
        
        # Calculate epipolar error for verification
        epipolar_error = self._compute_epipolar_error(pts1, pts2, F, mask)
        
        # Check motion in masked region if provided
        if motion_mask is not None:
            motion_consistency = self._check_motion_consistency(
                pts1, pts2, motion_mask, mask
            )
        else:
            motion_consistency = 1.0
        
        # Compute motion magnitude
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]
        
        if len(inlier_pts1) > 0:
            motion_vectors = inlier_pts2 - inlier_pts1
            motion_magnitudes = np.linalg.norm(motion_vectors, axis=1)
            avg_motion = np.mean(motion_magnitudes)
            is_moving = avg_motion > self.motion_threshold
        else:
            avg_motion = 0.0
            is_moving = False
        
        # Calculate overall confidence
        geometric_confidence = inlier_ratio * motion_consistency
        
        return MotionHypothesis(
            is_moving=is_moving,
            confidence=geometric_confidence,
            epipolar_error=epipolar_error,
            feature_count=int(inliers),
            geometric_consistency=motion_consistency
        )
    
    def _compute_epipolar_error(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        F: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """Compute average epipolar error for inliers."""
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]
        
        if len(inlier_pts1) == 0:
            return float('inf')
        
        # Convert to homogeneous coordinates
        pts1_h = np.hstack([inlier_pts1, np.ones((len(inlier_pts1), 1))])
        pts2_h = np.hstack([inlier_pts2, np.ones((len(inlier_pts2), 1))])
        
        # Compute epipolar error: x2^T * F * x1
        errors = np.abs(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))
        avg_error = np.mean(errors)
        
        return float(avg_error)
    
    def _check_motion_consistency(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        motion_mask: np.ndarray,
        inlier_mask: np.ndarray
    ) -> float:
        """
        Check consistency between geometric motion and detection mask.
        
        Returns:
            Consistency score between 0 and 1
        """
        inlier_pts1 = pts1[inlier_mask.ravel() == 1]
        inlier_pts2 = pts2[inlier_mask.ravel() == 1]
        
        if len(inlier_pts1) == 0:
            return 0.0
        
        # Count how many moving features are in motion mask
        moving_in_mask = 0
        total_moving = 0
        
        for p1, p2 in zip(inlier_pts1, inlier_pts2):
            motion = np.linalg.norm(p2 - p1)
            if motion > self.motion_threshold:
                total_moving += 1
                x, y = int(p2[0]), int(p2[1])
                if 0 <= y < motion_mask.shape[0] and 0 <= x < motion_mask.shape[1]:
                    if motion_mask[y, x] > 0:
                        moving_in_mask += 1
        
        if total_moving == 0:
            return 1.0
        
        return moving_in_mask / total_moving
    
    def reset(self):
        """Reset the verifier state."""
        self.prev_keypoints = None
        self.prev_descriptors = None
