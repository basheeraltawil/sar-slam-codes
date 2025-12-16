"""Unit tests for MovingObjectDetector."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sar_slam.moving_object_detector import MovingObjectDetector


class TestMovingObjectDetector(unittest.TestCase):
    """Test cases for MovingObjectDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = MovingObjectDetector(
            flow_threshold=2.0,
            max_corners=500
        )
        self.frame_shape = (480, 640)
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.flow_threshold, 2.0)
        self.assertEqual(self.detector.max_corners, 500)
    
    def test_detect_first_frame(self):
        """Test detection on first frame."""
        frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        
        motion_mask, moving_features, confidence = self.detector.detect(frame)
        
        # First frame should have no motion
        self.assertEqual(motion_mask.shape, self.frame_shape)
        self.assertEqual(len(moving_features), 0)
        self.assertEqual(confidence, 0.0)
    
    def test_detect_static_scene(self):
        """Test detection with static scene."""
        frame1 = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        frame2 = frame1.copy()
        
        self.detector.detect(frame1)
        motion_mask, moving_features, confidence = self.detector.detect(frame2)
        
        # Static scene should have low confidence
        self.assertLessEqual(confidence, 0.3)
    
    def test_detect_with_mask(self):
        """Test detection with region mask."""
        frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        mask = np.zeros(self.frame_shape, dtype=np.uint8)
        mask[100:300, 200:400] = 255
        
        motion_mask, moving_features, confidence = self.detector.detect(frame, mask)
        
        self.assertEqual(motion_mask.shape, self.frame_shape)
    
    def test_reset(self):
        """Test detector reset."""
        frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        self.detector.detect(frame)
        
        self.assertIsNotNone(self.detector.prev_gray)
        
        self.detector.reset()
        
        self.assertIsNone(self.detector.prev_gray)
        self.assertIsNone(self.detector.prev_features)
    
    def test_color_frame(self):
        """Test detection with color frame."""
        frame = np.random.randint(0, 255, (*self.frame_shape, 3), dtype=np.uint8)
        
        motion_mask, moving_features, confidence = self.detector.detect(frame)
        
        self.assertEqual(motion_mask.shape, self.frame_shape)


if __name__ == '__main__':
    unittest.main()
