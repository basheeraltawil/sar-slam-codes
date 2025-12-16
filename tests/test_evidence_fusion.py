"""Unit tests for EvidenceFusion."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sar_slam.evidence_fusion import EvidenceFusion, FusedEvidence, ObjectState


class TestEvidenceFusion(unittest.TestCase):
    """Test cases for EvidenceFusion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fusion = EvidenceFusion(
            detection_weight=0.4,
            geometric_weight=0.6,
            static_prior=0.7
        )
    
    def test_initialization(self):
        """Test fusion initialization."""
        self.assertIsNotNone(self.fusion)
        self.assertAlmostEqual(self.fusion.detection_weight + self.fusion.geometric_weight, 1.0)
    
    def test_fuse_high_confidence(self):
        """Test fusion with high confidence evidence."""
        evidence = self.fusion.fuse(
            detection_confidence=0.9,
            geometric_confidence=0.8,
            geometric_consistency=0.9
        )
        
        self.assertIsInstance(evidence, FusedEvidence)
        self.assertEqual(evidence.state, ObjectState.DYNAMIC)
        self.assertGreater(evidence.confidence, 0.5)
        self.assertFalse(evidence.should_track)
    
    def test_fuse_low_confidence(self):
        """Test fusion with low confidence evidence."""
        evidence = self.fusion.fuse(
            detection_confidence=0.1,
            geometric_confidence=0.2,
            geometric_consistency=0.3
        )
        
        self.assertIsInstance(evidence, FusedEvidence)
        self.assertEqual(evidence.state, ObjectState.STATIC)
        self.assertLess(evidence.confidence, 0.5)
        self.assertTrue(evidence.should_track)
    
    def test_fuse_uncertain(self):
        """Test fusion with uncertain evidence."""
        evidence = self.fusion.fuse(
            detection_confidence=0.5,
            geometric_confidence=0.4,
            geometric_consistency=0.5
        )
        
        self.assertIsInstance(evidence, FusedEvidence)
        self.assertIn(evidence.state, [ObjectState.STATIC, ObjectState.UNCERTAIN, ObjectState.DYNAMIC])
    
    def test_temporal_consistency(self):
        """Test temporal consistency tracking."""
        # Add multiple measurements
        for i in range(5):
            confidence = 0.7 + i * 0.05
            self.fusion.fuse(confidence, confidence, 0.9)
        
        stats = self.fusion.get_statistics()
        self.assertGreater(stats['temporal_consistency'], 0.0)
        self.assertEqual(stats['history_length'], 5)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        self.fusion.fuse(0.5, 0.6, 0.7)
        
        stats = self.fusion.get_statistics()
        
        self.assertIn('mean_score', stats)
        self.assertIn('std_score', stats)
        self.assertIn('temporal_consistency', stats)
        self.assertIn('history_length', stats)
    
    def test_reset(self):
        """Test fusion reset."""
        self.fusion.fuse(0.5, 0.6, 0.7)
        self.assertGreater(len(self.fusion.history), 0)
        
        self.fusion.reset()
        
        self.assertEqual(len(self.fusion.history), 0)
    
    def test_bayesian_update(self):
        """Test Bayesian update logic."""
        # High likelihood should increase posterior
        posterior1 = self.fusion._bayesian_update(0.9, 0.7)
        self.assertGreater(posterior1, 0.3)
        
        # Low likelihood should decrease posterior
        posterior2 = self.fusion._bayesian_update(0.1, 0.7)
        self.assertAlmostEqual(posterior2, 0.3, places=1)


if __name__ == '__main__':
    unittest.main()
