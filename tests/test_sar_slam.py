"""Unit tests for SARSLAM main pipeline."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sar_slam.sar_slam import SARSLAM, SARSLAMResult


class TestSARSLAM(unittest.TestCase):
    """Test cases for SARSLAM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.slam = SARSLAM(
            flow_threshold=2.0,
            ransac_threshold=1.0,
            detection_weight=0.4,
            geometric_weight=0.6
        )
        self.frame_shape = (480, 640, 3)
    
    def test_initialization(self):
        """Test SLAM initialization."""
        self.assertIsNotNone(self.slam)
        self.assertIsNotNone(self.slam.detector)
        self.assertIsNotNone(self.slam.verifier)
        self.assertIsNotNone(self.slam.fusion)
    
    def test_process_first_frame(self):
        """Test processing first frame."""
        frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        
        result = self.slam.process_frame(frame)
        
        self.assertIsInstance(result, SARSLAMResult)
        self.assertEqual(result.motion_mask.shape, self.frame_shape[:2])
        self.assertEqual(result.static_mask.shape, self.frame_shape[:2])
    
    def test_process_multiple_frames(self):
        """Test processing multiple frames."""
        for i in range(5):
            frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
            result = self.slam.process_frame(frame)
            
            self.assertIsInstance(result, SARSLAMResult)
            self.assertGreaterEqual(result.tracking_quality, 0.0)
            self.assertLessEqual(result.tracking_quality, 1.0)
    
    def test_process_with_mask(self):
        """Test processing with region mask."""
        frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        mask[100:300, 200:400] = 255
        
        result = self.slam.process_frame(frame, mask)
        
        self.assertIsInstance(result, SARSLAMResult)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        self.slam.process_frame(frame)
        
        stats = self.slam.get_statistics()
        
        self.assertIn('total_frames', stats)
        self.assertIn('static_frames', stats)
        self.assertIn('dynamic_frames', stats)
        self.assertIn('uncertain_frames', stats)
        self.assertIn('fusion', stats)
    
    def test_visualize(self):
        """Test visualization."""
        frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        result = self.slam.process_frame(frame)
        
        vis_frame = self.slam.visualize(frame, result)
        
        self.assertEqual(vis_frame.shape, self.frame_shape)
        self.assertEqual(vis_frame.dtype, np.uint8)
    
    def test_visualize_grayscale(self):
        """Test visualization with grayscale frame."""
        frame = np.random.randint(0, 255, self.frame_shape[:2], dtype=np.uint8)
        result = self.slam.process_frame(frame)
        
        vis_frame = self.slam.visualize(frame, result)
        
        self.assertEqual(len(vis_frame.shape), 3)
        self.assertEqual(vis_frame.shape[2], 3)
    
    def test_reset(self):
        """Test SLAM reset."""
        frame = np.random.randint(0, 255, self.frame_shape, dtype=np.uint8)
        self.slam.process_frame(frame)
        
        self.assertGreater(self.slam.frame_count, 0)
        
        self.slam.reset()
        
        self.assertEqual(self.slam.frame_count, 0)
        self.assertIsNone(self.slam.prev_frame)
    
    def test_static_mask_creation(self):
        """Test static mask creation."""
        motion_mask = np.zeros((480, 640), dtype=np.uint8)
        motion_mask[100:200, 100:200] = 255
        
        from sar_slam.evidence_fusion import FusedEvidence, ObjectState
        
        # Test with static state
        evidence = FusedEvidence(
            state=ObjectState.STATIC,
            confidence=0.9,
            detection_score=0.1,
            geometric_score=0.1,
            final_score=0.1,
            should_track=True
        )
        
        static_mask = self.slam._create_static_mask(motion_mask, evidence)
        self.assertEqual(static_mask.shape, motion_mask.shape)
    
    def test_tracking_quality_computation(self):
        """Test tracking quality computation."""
        from sar_slam.evidence_fusion import FusedEvidence, ObjectState
        from sar_slam.geometric_verifier import MotionHypothesis
        
        evidence = FusedEvidence(
            state=ObjectState.STATIC,
            confidence=0.8,
            detection_score=0.2,
            geometric_score=0.7,
            final_score=0.5,
            should_track=True
        )
        
        hypothesis = MotionHypothesis(
            is_moving=False,
            confidence=0.7,
            epipolar_error=2.5,
            feature_count=50,
            geometric_consistency=0.8
        )
        
        quality = self.slam._compute_tracking_quality(evidence, hypothesis)
        
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)


if __name__ == '__main__':
    unittest.main()
