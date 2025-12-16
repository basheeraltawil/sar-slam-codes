"""
Basic Usage Example

Demonstrates basic usage of SAR-SLAM components.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sar_slam import (
    MovingObjectDetector,
    GeometricMotionVerifier,
    EvidenceFusion,
    SARSLAM
)


def example_detector():
    """Example: Using MovingObjectDetector alone."""
    print("=== Moving Object Detector Example ===")
    
    detector = MovingObjectDetector(
        flow_threshold=2.0,
        max_corners=500
    )
    
    # Simulate two frames
    frame1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    # Detect motion
    motion_mask, moving_features, confidence = detector.detect(frame1)
    print(f"Frame 1 - Moving features: {len(moving_features)}, Confidence: {confidence:.2f}")
    
    motion_mask, moving_features, confidence = detector.detect(frame2)
    print(f"Frame 2 - Moving features: {len(moving_features)}, Confidence: {confidence:.2f}")
    
    print()


def example_verifier():
    """Example: Using GeometricMotionVerifier alone."""
    print("=== Geometric Motion Verifier Example ===")
    
    verifier = GeometricMotionVerifier(
        ransac_threshold=1.0,
        min_inliers=8
    )
    
    # Simulate two frames
    frame1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    # Verify motion
    hypothesis = verifier.verify(frame1, frame2)
    
    print(f"Is moving: {hypothesis.is_moving}")
    print(f"Confidence: {hypothesis.confidence:.2f}")
    print(f"Feature count: {hypothesis.feature_count}")
    print(f"Geometric consistency: {hypothesis.geometric_consistency:.2f}")
    print(f"Epipolar error: {hypothesis.epipolar_error:.2f}")
    
    print()


def example_fusion():
    """Example: Using EvidenceFusion alone."""
    print("=== Evidence Fusion Example ===")
    
    fusion = EvidenceFusion(
        detection_weight=0.4,
        geometric_weight=0.6,
        static_prior=0.7
    )
    
    # Simulate evidence from multiple frames
    scenarios = [
        (0.8, 0.7, 0.9, "High motion detected and verified"),
        (0.2, 0.3, 0.5, "Low motion, likely static"),
        (0.6, 0.4, 0.6, "Moderate motion, uncertain"),
    ]
    
    for det_conf, geo_conf, geo_cons, description in scenarios:
        evidence = fusion.fuse(det_conf, geo_conf, geo_cons)
        
        print(f"\nScenario: {description}")
        print(f"  Detection confidence: {det_conf:.2f}")
        print(f"  Geometric confidence: {geo_conf:.2f}")
        print(f"  Geometric consistency: {geo_cons:.2f}")
        print(f"  → State: {evidence.state.name}")
        print(f"  → Final confidence: {evidence.confidence:.2f}")
        print(f"  → Should track: {evidence.should_track}")
    
    print()


def example_full_pipeline():
    """Example: Using full SAR-SLAM pipeline."""
    print("=== Full SAR-SLAM Pipeline Example ===")
    
    slam = SARSLAM(
        flow_threshold=2.0,
        ransac_threshold=1.0,
        detection_weight=0.4,
        geometric_weight=0.6
    )
    
    # Simulate processing multiple frames
    print("\nProcessing synthetic frames...")
    
    for i in range(5):
        # Create synthetic frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process frame
        result = slam.process_frame(frame)
        
        print(f"\nFrame {i+1}:")
        print(f"  State: {result.fused_evidence.state.name}")
        print(f"  Confidence: {result.fused_evidence.confidence:.2f}")
        print(f"  Tracking quality: {result.tracking_quality:.2f}")
        print(f"  Moving features: {len(result.moving_features)}")
    
    # Get final statistics
    print("\n=== Statistics ===")
    stats = slam.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


def main():
    """Run all examples."""
    print("SAR-SLAM Basic Usage Examples")
    print("=" * 50)
    print()
    
    example_detector()
    example_verifier()
    example_fusion()
    example_full_pipeline()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == '__main__':
    main()
