"""
SAR-SLAM: Bio-inspired SLAM for Dynamic Environments

Main pipeline that integrates moving object detection, geometric verification,
and evidence fusion for robust SLAM in dynamic environments.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from .moving_object_detector import MovingObjectDetector
from .geometric_verifier import GeometricMotionVerifier, MotionHypothesis
from .evidence_fusion import EvidenceFusion, FusedEvidence, ObjectState


@dataclass
class SARSLAMResult:
    """Result from SAR-SLAM processing."""
    motion_mask: np.ndarray
    static_mask: np.ndarray
    fused_evidence: FusedEvidence
    geometric_hypothesis: MotionHypothesis
    moving_features: List[np.ndarray]
    tracking_quality: float


class SARSLAM:
    """
    Bio-inspired SLAM system for dynamic environments.
    
    Processing pipeline:
    1. Identify moving objects using visual cues
    2. Verify motion using geometric constraints
    3. Fuse evidence intelligently for robust tracking
    """
    
    def __init__(
        self,
        flow_threshold: float = 2.0,
        ransac_threshold: float = 1.0,
        detection_weight: float = 0.4,
        geometric_weight: float = 0.6,
        static_prior: float = 0.7,
        enable_visualization: bool = False
    ):
        """
        Initialize SAR-SLAM system.
        
        Args:
            flow_threshold: Threshold for optical flow motion detection
            ransac_threshold: RANSAC threshold for geometric verification
            detection_weight: Weight for detection evidence
            geometric_weight: Weight for geometric evidence
            static_prior: Prior probability of static scene
            enable_visualization: Enable visualization features
        """
        # Initialize modules
        self.detector = MovingObjectDetector(flow_threshold=flow_threshold)
        self.verifier = GeometricMotionVerifier(ransac_threshold=ransac_threshold)
        self.fusion = EvidenceFusion(
            detection_weight=detection_weight,
            geometric_weight=geometric_weight,
            static_prior=static_prior
        )
        
        self.enable_visualization = enable_visualization
        self.prev_frame = None
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'dynamic_frames': 0,
            'static_frames': 0,
            'uncertain_frames': 0
        }
    
    def process_frame(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> SARSLAMResult:
        """
        Process a single frame through the SAR-SLAM pipeline.
        
        Args:
            frame: Input frame (BGR or grayscale)
            mask: Optional region mask
            
        Returns:
            SARSLAMResult with processing results
        """
        self.frame_count += 1
        
        # Step 1: Identify moving objects
        motion_mask, moving_features, detection_confidence = self.detector.detect(
            frame, mask
        )
        
        # Initialize result for first frame
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            static_mask = np.ones_like(motion_mask) * 255
            
            # Create default results
            default_hypothesis = MotionHypothesis(
                is_moving=False,
                confidence=0.0,
                epipolar_error=0.0,
                feature_count=0,
                geometric_consistency=1.0
            )
            
            default_evidence = FusedEvidence(
                state=ObjectState.STATIC,
                confidence=1.0,
                detection_score=0.0,
                geometric_score=0.0,
                final_score=0.0,
                should_track=True
            )
            
            return SARSLAMResult(
                motion_mask=motion_mask,
                static_mask=static_mask,
                fused_evidence=default_evidence,
                geometric_hypothesis=default_hypothesis,
                moving_features=moving_features,
                tracking_quality=1.0
            )
        
        # Step 2: Verify motion geometrically
        geometric_hypothesis = self.verifier.verify(
            self.prev_frame, frame, motion_mask
        )
        
        # Step 3: Fuse evidence intelligently
        fused_evidence = self.fusion.fuse(
            detection_confidence=detection_confidence,
            geometric_confidence=geometric_hypothesis.confidence,
            geometric_consistency=geometric_hypothesis.geometric_consistency,
            use_temporal=True
        )
        
        # Create static mask (inverse of motion for tracking)
        static_mask = self._create_static_mask(motion_mask, fused_evidence)
        
        # Calculate tracking quality
        tracking_quality = self._compute_tracking_quality(
            fused_evidence, geometric_hypothesis
        )
        
        # Update statistics
        self._update_statistics(fused_evidence.state)
        
        # Store current frame
        self.prev_frame = frame.copy()
        
        return SARSLAMResult(
            motion_mask=motion_mask,
            static_mask=static_mask,
            fused_evidence=fused_evidence,
            geometric_hypothesis=geometric_hypothesis,
            moving_features=moving_features,
            tracking_quality=tracking_quality
        )
    
    def _create_static_mask(
        self,
        motion_mask: np.ndarray,
        evidence: FusedEvidence
    ) -> np.ndarray:
        """
        Create mask of static regions suitable for SLAM tracking.
        
        Args:
            motion_mask: Binary mask of detected motion
            evidence: Fused evidence result
            
        Returns:
            Static mask for tracking
        """
        if evidence.state == ObjectState.STATIC:
            # Entire scene is static
            return np.ones_like(motion_mask) * 255
        elif evidence.state == ObjectState.DYNAMIC:
            # Exclude moving regions
            static_mask = cv2.bitwise_not(motion_mask)
            # Erode slightly to ensure clean boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            static_mask = cv2.erode(static_mask, kernel, iterations=1)
            return static_mask
        else:
            # Uncertain - use conservative approach
            # Slightly expand motion mask to be safe
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            expanded_motion = cv2.dilate(motion_mask, kernel, iterations=1)
            static_mask = cv2.bitwise_not(expanded_motion)
            return static_mask
    
    def _compute_tracking_quality(
        self,
        evidence: FusedEvidence,
        hypothesis: MotionHypothesis
    ) -> float:
        """
        Compute overall tracking quality metric.
        
        Returns:
            Quality score between 0 and 1
        """
        # Factors affecting tracking quality
        confidence_factor = evidence.confidence
        feature_factor = min(hypothesis.feature_count / 50.0, 1.0)
        consistency_factor = hypothesis.geometric_consistency
        
        # Geometric error factor (lower error is better)
        if hypothesis.epipolar_error < float('inf'):
            error_factor = max(0, 1.0 - hypothesis.epipolar_error / 10.0)
        else:
            error_factor = 0.0
        
        # Weighted combination
        quality = (
            0.3 * confidence_factor +
            0.2 * feature_factor +
            0.3 * consistency_factor +
            0.2 * error_factor
        )
        
        return float(np.clip(quality, 0.0, 1.0))
    
    def _update_statistics(self, state: ObjectState):
        """Update frame statistics."""
        self.stats['total_frames'] += 1
        
        if state == ObjectState.STATIC:
            self.stats['static_frames'] += 1
        elif state == ObjectState.DYNAMIC:
            self.stats['dynamic_frames'] += 1
        else:
            self.stats['uncertain_frames'] += 1
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with system statistics
        """
        fusion_stats = self.fusion.get_statistics()
        
        return {
            **self.stats,
            'fusion': fusion_stats,
            'frames_processed': self.frame_count
        }
    
    def visualize(
        self,
        frame: np.ndarray,
        result: SARSLAMResult
    ) -> np.ndarray:
        """
        Create visualization of SAR-SLAM results.
        
        Args:
            frame: Original frame
            result: SAR-SLAM processing result
            
        Returns:
            Annotated frame
        """
        if len(frame.shape) == 2:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis_frame = frame.copy()
        
        # Overlay motion mask
        motion_overlay = np.zeros_like(vis_frame)
        motion_overlay[:, :, 2] = result.motion_mask  # Red for motion
        vis_frame = cv2.addWeighted(vis_frame, 0.7, motion_overlay, 0.3, 0)
        
        # Overlay static mask
        static_overlay = np.zeros_like(vis_frame)
        static_overlay[:, :, 1] = result.static_mask  # Green for static
        vis_frame = cv2.addWeighted(vis_frame, 0.8, static_overlay, 0.2, 0)
        
        # Draw moving features
        for feature in result.moving_features:
            if len(feature) > 0:
                x, y = int(feature[0]), int(feature[1])
                cv2.circle(vis_frame, (x, y), 5, (255, 0, 255), 2)
        
        # Add text information
        state_colors = {
            ObjectState.STATIC: (0, 255, 0),
            ObjectState.DYNAMIC: (0, 0, 255),
            ObjectState.UNCERTAIN: (0, 165, 255)
        }
        
        state_text = f"State: {result.fused_evidence.state.name}"
        confidence_text = f"Confidence: {result.fused_evidence.confidence:.2f}"
        quality_text = f"Tracking Quality: {result.tracking_quality:.2f}"
        
        color = state_colors[result.fused_evidence.state]
        cv2.putText(vis_frame, state_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis_frame, confidence_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, quality_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def reset(self):
        """Reset the entire SAR-SLAM system."""
        self.detector.reset()
        self.verifier.reset()
        self.fusion.reset()
        self.prev_frame = None
        self.frame_count = 0
        self.stats = {
            'total_frames': 0,
            'dynamic_frames': 0,
            'static_frames': 0,
            'uncertain_frames': 0
        }
