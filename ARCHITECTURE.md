# SAR-SLAM Architecture

## System Overview

SAR-SLAM implements a bio-inspired three-stage pipeline for robust SLAM in dynamic environments:

```
Input Frame → Moving Object Detection → Geometric Verification → Evidence Fusion → Output (Masks + State)
```

## Module Architecture

### 1. Moving Object Detector (`moving_object_detector.py`)

**Purpose**: Identify potentially moving objects using visual cues

**Key Algorithms**:
- **Feature Detection**: Shi-Tomasi corner detection
  - Detects prominent corners in the image
  - Quality-based filtering ensures robust features
  
- **Optical Flow**: Lucas-Kanade pyramidal method
  - Tracks features between consecutive frames
  - Pyramidal approach handles large motions
  
- **Motion Analysis**: Flow magnitude thresholding
  - Computes motion vectors from feature displacement
  - Thresholding separates moving from static features

**Outputs**:
- Binary motion mask
- List of moving feature points
- Detection confidence score

### 2. Geometric Motion Verifier (`geometric_verifier.py`)

**Purpose**: Verify motion using geometric constraints and 3D structure

**Key Algorithms**:
- **Feature Matching**: ORB features with BFMatcher
  - Scale and rotation invariant descriptors
  - Efficient binary descriptor matching
  
- **Outlier Rejection**: Lowe's ratio test
  - Filters ambiguous matches
  - Improves geometric estimation quality
  
- **Geometry Estimation**: RANSAC-based fundamental matrix
  - Robust to outliers
  - Recovers epipolar geometry between frames
  
- **Consistency Checking**: Epipolar error analysis
  - Validates motion hypothesis
  - Measures geometric consistency

**Outputs**:
- Motion hypothesis with geometric evidence
- Inlier count and confidence
- Epipolar error metric
- Geometric consistency score

### 3. Evidence Fusion (`evidence_fusion.py`)

**Purpose**: Intelligently fuse evidence from detection and verification

**Key Algorithms**:
- **Bayesian Inference**:
  ```
  P(dynamic | evidence) = P(evidence | dynamic) * P(dynamic) / P(evidence)
  ```
  - Updates belief based on observed evidence
  - Incorporates prior knowledge about scene statistics
  
- **Temporal Filtering**: Exponentially weighted moving average
  - Smooths evidence over time
  - Reduces false positives from noise
  
- **State Classification**:
  - STATIC: High confidence in static scene
  - DYNAMIC: High confidence in motion
  - UNCERTAIN: Ambiguous evidence

**Outputs**:
- Object state classification
- Fused confidence score
- Tracking recommendation
- Temporal statistics

## Data Flow

```
Frame(t-1) ──┐
             ├─→ [Moving Object Detector] ──→ Motion Mask
Frame(t) ────┤                               + Moving Features
             └─→ [Geometric Verifier] ──────→ Motion Hypothesis
                                              
Motion Mask + Motion Hypothesis
             │
             ├─→ [Evidence Fusion] ──────────→ Final State
             │                                + Static Mask
             └──────────────────────────────→ + Confidence
```

## Integration with SLAM Systems

SAR-SLAM can be integrated into existing SLAM pipelines:

### Traditional SLAM Pipeline:
```
Feature Extraction → Feature Matching → Pose Estimation → Mapping
```

### Enhanced SAR-SLAM Pipeline:
```
Feature Extraction → [SAR-SLAM: Filter Dynamic Objects] → Feature Matching → Pose Estimation → Mapping
                                     ↓
                              Static Features Only
```

### Usage Pattern:

```python
# Initialize SAR-SLAM
sar_slam = SARSLAM()

# In your SLAM loop
for frame in video_stream:
    # Process with SAR-SLAM
    result = sar_slam.process_frame(frame)
    
    # Use static mask to filter features
    static_features = extract_features(frame, mask=result.static_mask)
    
    # Continue with traditional SLAM
    matches = match_features(static_features, map_points)
    pose = estimate_pose(matches)
    update_map(pose, static_features)
```

## Performance Characteristics

### Computational Complexity

| Module | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| Moving Object Detector | O(n) | O(n) |
| Geometric Verifier | O(n log n) | O(n) |
| Evidence Fusion | O(1) | O(h) |

Where:
- n = number of features
- h = history length (default: 10)

### Typical Performance

On a standard CPU (Intel i7):
- **480x640 resolution**: ~30 FPS
- **720p resolution**: ~20 FPS
- **1080p resolution**: ~10 FPS

GPU acceleration possible through OpenCV's CUDA support.

## Parameter Tuning Guide

### For Indoor Environments:
```python
SARSLAM(
    flow_threshold=1.5,      # Lower for slower motions
    ransac_threshold=0.5,    # Tighter for structured scenes
    static_prior=0.8         # Higher prior for static scenes
)
```

### For Outdoor Environments:
```python
SARSLAM(
    flow_threshold=3.0,      # Higher for faster motions
    ransac_threshold=1.5,    # Looser for unstructured scenes
    static_prior=0.6         # Lower prior, more dynamic
)
```

### For Crowded Scenes:
```python
SARSLAM(
    flow_threshold=2.0,
    detection_weight=0.5,    # Increase detection influence
    geometric_weight=0.5,
    static_prior=0.5         # Balanced prior
)
```

## Design Principles

1. **Modularity**: Each component can be used independently
2. **Robustness**: Multiple evidence sources reduce false positives
3. **Efficiency**: Optimized for real-time performance
4. **Adaptability**: Parameters can be tuned for different scenarios
5. **Transparency**: Clear outputs and statistics for debugging

## Future Extensions

Potential enhancements:
- Deep learning-based object detection
- Multi-frame temporal analysis
- Semantic understanding of dynamic objects
- Adaptive parameter tuning
- GPU acceleration
- IMU integration for motion prediction
