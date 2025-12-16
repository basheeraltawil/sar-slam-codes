"""
SAR-SLAM: Bio-inspired SLAM for Dynamic Environments

This package implements a bio-inspired approach to SLAM that handles dynamic scenes
by processing them like the human visual system:
1. Identify moving objects
2. Verify motion geometrically
3. Intelligently fuse evidence for robust tracking
"""

from .moving_object_detector import MovingObjectDetector
from .geometric_verifier import GeometricMotionVerifier
from .evidence_fusion import EvidenceFusion
from .sar_slam import SARSLAM

__version__ = '0.1.0'
__all__ = [
    'MovingObjectDetector',
    'GeometricMotionVerifier',
    'EvidenceFusion',
    'SARSLAM',
]
