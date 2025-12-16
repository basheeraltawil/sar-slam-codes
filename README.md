# SAR-SLAM: Bio-inspired SLAM for Dynamic Environments

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

SAR-SLAM (Scene-Aware Robust SLAM) is a bio-inspired approach to Simultaneous Localization and Mapping (SLAM) designed to handle dynamic environments. Traditional SLAM systems assume a static world, but real-world environments are inherently dynamic with people walking, vehicles moving, and objects shifting.

## Overview

SAR-SLAM processes dynamic scenes similar to the human visual system through a three-stage pipeline:

1. **Moving Object Detection** - First identifies potentially moving objects using visual cues
2. **Geometric Motion Verification** - Verifies actual motion using geometric constraints
3. **Intelligent Evidence Fusion** - Fuses both sources of evidence to maintain robust tracking

This bio-inspired approach enables robust SLAM performance in human-centric environments where traditional methods struggle.

## Key Features

- **Bio-inspired Architecture**: Mimics human visual processing for scene understanding
- **Multi-modal Evidence**: Combines visual detection with geometric verification
- **Bayesian Fusion**: Intelligently fuses evidence with temporal consistency
- **Robust Tracking**: Maintains accurate localization in dynamic scenes
- **Modular Design**: Each component can be used independently
- **Real-time Capable**: Optimized for efficient processing
- **Easy Integration**: Simple API for integration into existing SLAM systems

## Installation

### Requirements

- Python 3.7 or higher
- NumPy >= 1.19.0
- OpenCV >= 4.5.0
- SciPy >= 1.5.0
- Matplotlib >= 3.3.0

### Install from Source

```bash
git clone https://github.com/basheeraltawil/sar-slam-codes.git
cd sar-slam-codes
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Basic Usage

```python
from sar_slam import SARSLAM
import cv2

# Initialize SAR-SLAM
slam = SARSLAM(
    flow_threshold=2.0,
    ransac_threshold=1.0,
    detection_weight=0.4,
    geometric_weight=0.6
)

# Open video source
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    result = slam.process_frame(frame)
    
    # Access results
    print(f"State: {result.fused_evidence.state.name}")
    print(f"Tracking Quality: {result.tracking_quality:.2f}")
    
    # Visualize
    vis_frame = slam.visualize(frame, result)
    cv2.imshow('SAR-SLAM', vis_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Architecture

### 1. Moving Object Detector

Identifies potentially moving objects using:
- Optical flow analysis (Lucas-Kanade)
- Feature tracking (Shi-Tomasi corners)
- Motion pattern analysis

```python
from sar_slam import MovingObjectDetector

detector = MovingObjectDetector(flow_threshold=2.0)
motion_mask, moving_features, confidence = detector.detect(frame)
```

### 2. Geometric Motion Verifier

Verifies motion using geometric constraints:
- Epipolar geometry
- Fundamental matrix estimation (RANSAC)
- 3D motion consistency checks

```python
from sar_slam import GeometricMotionVerifier

verifier = GeometricMotionVerifier(ransac_threshold=1.0)
hypothesis = verifier.verify(frame1, frame2, motion_mask)
```

### 3. Evidence Fusion

Intelligently fuses evidence using:
- Bayesian inference
- Temporal consistency tracking
- Adaptive confidence weighting

```python
from sar_slam import EvidenceFusion

fusion = EvidenceFusion(
    detection_weight=0.4,
    geometric_weight=0.6
)
evidence = fusion.fuse(
    detection_confidence,
    geometric_confidence,
    geometric_consistency
)
```

## Examples

The `examples/` directory contains demonstrations:

### Video Processing

```bash
# Process video file
python examples/demo_video.py video.mp4 --output result.mp4

# Use webcam
python examples/demo_video.py 0
```

### Image Sequences

```bash
# Process image directory
python examples/demo_images.py /path/to/images --output /path/to/output
```

### Basic Usage Examples

```bash
# Run basic examples
python examples/basic_usage.py
```

## Configuration

### SAR-SLAM Parameters

```python
slam = SARSLAM(
    flow_threshold=2.0,        # Optical flow threshold for motion
    ransac_threshold=1.0,      # RANSAC reprojection threshold
    detection_weight=0.4,      # Weight for detection evidence
    geometric_weight=0.6,      # Weight for geometric evidence
    static_prior=0.7,          # Prior probability of static scene
    enable_visualization=True  # Enable visualization features
)
```

### Detector Parameters

```python
detector = MovingObjectDetector(
    flow_threshold=2.0,        # Minimum flow magnitude
    min_feature_distance=10,   # Min distance between features
    max_corners=500,           # Maximum corners to track
    quality_level=0.01         # Corner detection quality
)
```

### Verifier Parameters

```python
verifier = GeometricMotionVerifier(
    ransac_threshold=1.0,      # RANSAC threshold
    min_inliers=8,             # Minimum inliers required
    confidence=0.99,           # RANSAC confidence level
    motion_threshold=2.0       # Motion magnitude threshold
)
```

### Fusion Parameters

```python
fusion = EvidenceFusion(
    detection_weight=0.4,          # Detection evidence weight
    geometric_weight=0.6,          # Geometric evidence weight
    static_prior=0.7,              # Prior for static scenes
    confidence_threshold=0.5,      # Classification threshold
    uncertainty_threshold=0.3      # Uncertainty threshold
)
```

## Use Cases

SAR-SLAM is designed for:

- **Autonomous Navigation**: Robots navigating crowded spaces
- **AR/VR Applications**: Mixed reality in dynamic environments
- **Surveillance Systems**: Tracking in scenes with moving objects
- **Autonomous Vehicles**: Urban driving with pedestrians and traffic
- **Service Robots**: Indoor navigation with human interaction
- **Drone Navigation**: Flying in dynamic outdoor scenes

## Technical Details

### Motion Detection Pipeline

1. **Feature Detection**: Shi-Tomasi corner detection
2. **Optical Flow**: Lucas-Kanade pyramidal method
3. **Motion Analysis**: Flow magnitude thresholding
4. **Region Extraction**: Morphological operations for coherent regions

### Geometric Verification

1. **Feature Matching**: ORB features with BFMatcher
2. **Outlier Rejection**: Lowe's ratio test
3. **Geometry Estimation**: RANSAC-based fundamental matrix
4. **Consistency Check**: Epipolar error analysis

### Evidence Fusion

1. **Bayesian Update**: P(dynamic | evidence)
2. **Temporal Filtering**: Exponentially weighted moving average
3. **State Classification**: Static, Dynamic, or Uncertain
4. **Tracking Decision**: Mask generation for SLAM

## Performance Considerations

- **Real-time Processing**: Optimized for 30+ FPS on modern hardware
- **Memory Efficient**: Minimal memory footprint
- **Scalable**: Adjustable parameters for speed/accuracy tradeoff
- **Parallel-safe**: Can process multiple streams independently

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use SAR-SLAM in your research, please cite:

```bibtex
@software{sar_slam_2024,
  title = {SAR-SLAM: Bio-inspired SLAM for Dynamic Environments},
  author = {Altawil, Basheer},
  year = {2024},
  url = {https://github.com/basheeraltawil/sar-slam-codes}
}
```

## Acknowledgments

This work is inspired by biological vision systems and builds upon classical computer vision and SLAM techniques.

## Contact

For questions and feedback:
- GitHub Issues: [https://github.com/basheeraltawil/sar-slam-codes/issues](https://github.com/basheeraltawil/sar-slam-codes/issues)