"""
Moving Object Detection Module

Bio-inspired moving object identification using visual features and motion patterns.
This module identifies potentially moving objects in the scene.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class MovingObjectDetector:
    """
    Detects moving objects in the scene using bio-inspired visual processing.
    
    This detector uses multiple cues to identify moving objects:
    - Optical flow analysis
    - Feature tracking
    - Appearance changes
    """
    
    def __init__(
        self,
        flow_threshold: float = 2.0,
        min_feature_distance: int = 10,
        max_corners: int = 500,
        quality_level: float = 0.01
    ):
        """
        Initialize the moving object detector.
        
        Args:
            flow_threshold: Minimum optical flow magnitude to consider motion
            min_feature_distance: Minimum distance between detected features
            max_corners: Maximum number of corners to track
            quality_level: Quality level for corner detection
        """
        self.flow_threshold = flow_threshold
        self.min_feature_distance = min_feature_distance
        self.max_corners = max_corners
        self.quality_level = quality_level
        
        self.prev_gray = None
        self.prev_features = None
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_feature_distance,
            blockSize=7
        )
    
    def detect(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray], float]:
        """
        Detect moving objects in the current frame.
        
        Args:
            frame: Current frame (BGR or grayscale)
            mask: Optional mask to restrict detection region
            
        Returns:
            Tuple containing:
            - Motion mask (binary image showing moving regions)
            - List of moving feature points
            - Confidence score for detection
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Initialize on first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_features = self._detect_features(gray, mask)
            return np.zeros_like(gray), [], 0.0
        
        # Detect features in current frame
        curr_features = self._detect_features(gray, mask)
        
        # Calculate optical flow
        flow_vectors, moving_features, confidence = self._compute_motion(
            self.prev_gray, gray, self.prev_features, curr_features
        )
        
        # Create motion mask
        motion_mask = self._create_motion_mask(gray.shape, moving_features, flow_vectors)
        
        # Update previous frame data
        self.prev_gray = gray
        self.prev_features = curr_features
        
        return motion_mask, moving_features, confidence
    
    def _detect_features(
        self,
        gray: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Detect corner features in the image."""
        features = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        return features if features is not None else np.array([])
    
    def _compute_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_features: np.ndarray,
        curr_features: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        Compute motion between frames using optical flow.
        
        Returns:
            Tuple of (flow_vectors, moving_features, confidence)
        """
        if len(prev_features) == 0:
            return [], [], 0.0
        
        # Calculate optical flow
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_features, None, **self.lk_params
        )
        
        if next_pts is None:
            return [], [], 0.0
        
        # Select good points
        good_prev = prev_features[status.flatten() == 1].reshape(-1, 2)
        good_next = next_pts[status.flatten() == 1].reshape(-1, 2)
        
        if len(good_prev) == 0:
            return [], [], 0.0
        
        # Compute flow vectors
        flow_vectors = good_next - good_prev
        flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
        
        # Identify moving features
        moving_mask = flow_magnitudes > self.flow_threshold
        moving_features = good_next[moving_mask]
        moving_flow = flow_vectors[moving_mask]
        
        # Calculate confidence based on consistency of motion
        if len(moving_features) > 0:
            confidence = len(moving_features) / len(good_prev)
        else:
            confidence = 0.0
        
        return moving_flow.tolist(), moving_features.tolist(), confidence
    
    def _create_motion_mask(
        self,
        shape: Tuple[int, int],
        moving_features: List[np.ndarray],
        flow_vectors: List[np.ndarray]
    ) -> np.ndarray:
        """Create a binary mask of moving regions."""
        mask = np.zeros(shape, dtype=np.uint8)
        
        for feature in moving_features:
            if len(feature) > 0:
                x, y = int(feature[0]), int(feature[1])
                cv2.circle(mask, (x, y), 15, 255, -1)
        
        # Dilate to connect nearby regions
        if len(moving_features) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def reset(self):
        """Reset the detector state."""
        self.prev_gray = None
        self.prev_features = None
