"""Unit tests for GeometricMotionVerifier."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sar_slam.geometric_verifier import GeometricMotionVerifier, MotionHypothesis


class TestGeometricMotionVerifier(unittest.TestCase):
    """Test cases for GeometricMotionVerifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = GeometricMotionVerifier(
            ransac_threshold=1.0,
            min_inliers=8
        )
        self.frame_shape = (480, 640)
    
    def test_initialization(self):
        """Test verifier initialization."""
        self.assertIsNotNone(self.verifier)
        self.assertEqual(self.verifier.ransac_threshold, 1.0)
        self.assertEqual(self.verifier.min_inliers, 8)
    
    def test_verify_basic(self):
        """Test basic verification."""
        frame1 = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        frame2 = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        
        hypothesis = self.verifier.verify(frame1, frame2)
        
        self.assertIsInstance(hypothesis, MotionHypothesis)
        self.assertIsInstance(hypothesis.is_moving, bool)
        self.assertGreaterEqual(hypothesis.confidence, 0.0)
        self.assertLessEqual(hypothesis.confidence, 1.0)
    
    def test_verify_identical_frames(self):
        """Test verification with identical frames."""
        frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        
        hypothesis = self.verifier.verify(frame, frame)
        
        # Identical frames should show no motion
        self.assertFalse(hypothesis.is_moving)
    
    def test_verify_with_mask(self):
        """Test verification with motion mask."""
        frame1 = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        frame2 = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        mask = np.zeros(self.frame_shape, dtype=np.uint8)
        mask[100:300, 200:400] = 255
        
        hypothesis = self.verifier.verify(frame1, frame2, mask)
        
        self.assertIsInstance(hypothesis, MotionHypothesis)
    
    def test_reset(self):
        """Test verifier reset."""
        self.verifier.reset()
        
        self.assertIsNone(self.verifier.prev_keypoints)
        self.assertIsNone(self.verifier.prev_descriptors)
    
    def test_color_frames(self):
        """Test verification with color frames."""
        frame1 = np.random.randint(0, 255, (*self.frame_shape, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (*self.frame_shape, 3), dtype=np.uint8)
        
        hypothesis = self.verifier.verify(frame1, frame2)
        
        self.assertIsInstance(hypothesis, MotionHypothesis)


if __name__ == '__main__':
    unittest.main()
