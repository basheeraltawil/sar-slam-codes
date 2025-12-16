"""
Evidence Fusion Module

Intelligently fuses evidence from moving object detection and geometric verification
to maintain robust tracking in dynamic environments.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ObjectState(Enum):
    """Possible states of tracked objects."""
    STATIC = 0
    DYNAMIC = 1
    UNCERTAIN = 2


@dataclass
class FusedEvidence:
    """Result of evidence fusion."""
    state: ObjectState
    confidence: float
    detection_score: float
    geometric_score: float
    final_score: float
    should_track: bool


class EvidenceFusion:
    """
    Fuses evidence from multiple sources using Bayesian inference.
    
    This module combines:
    - Visual motion detection evidence
    - Geometric verification evidence
    - Temporal consistency
    - Prior beliefs about the environment
    """
    
    def __init__(
        self,
        detection_weight: float = 0.4,
        geometric_weight: float = 0.6,
        static_prior: float = 0.7,
        confidence_threshold: float = 0.5,
        uncertainty_threshold: float = 0.3
    ):
        """
        Initialize the evidence fusion module.
        
        Args:
            detection_weight: Weight for detection evidence (0-1)
            geometric_weight: Weight for geometric evidence (0-1)
            static_prior: Prior probability that scene is static
            confidence_threshold: Threshold for classification
            uncertainty_threshold: Threshold for uncertain state
        """
        self.detection_weight = detection_weight
        self.geometric_weight = geometric_weight
        self.static_prior = static_prior
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Temporal tracking
        self.history = []
        self.max_history = 10
        
        # Normalize weights
        total_weight = detection_weight + geometric_weight
        self.detection_weight = detection_weight / total_weight
        self.geometric_weight = geometric_weight / total_weight
    
    def fuse(
        self,
        detection_confidence: float,
        geometric_confidence: float,
        geometric_consistency: float,
        use_temporal: bool = True
    ) -> FusedEvidence:
        """
        Fuse evidence from detection and geometric verification.
        
        Args:
            detection_confidence: Confidence from motion detection (0-1)
            geometric_confidence: Confidence from geometric verification (0-1)
            geometric_consistency: Consistency score from geometry (0-1)
            use_temporal: Whether to use temporal information
            
        Returns:
            FusedEvidence with classification result
        """
        # Weighted combination of evidence
        detection_score = detection_confidence
        geometric_score = geometric_confidence * geometric_consistency
        
        # Compute fused score using weighted average
        fused_score = (
            self.detection_weight * detection_score +
            self.geometric_weight * geometric_score
        )
        
        # Apply Bayesian update with prior
        posterior = self._bayesian_update(fused_score, self.static_prior)
        
        # Incorporate temporal consistency if enabled
        if use_temporal and len(self.history) > 0:
            temporal_score = self._compute_temporal_consistency()
            posterior = 0.7 * posterior + 0.3 * temporal_score
        
        # Update history
        self._update_history(fused_score)
        
        # Classify state
        state = self._classify_state(posterior)
        
        # Determine if object should be tracked
        should_track = self._should_track_object(state, posterior)
        
        return FusedEvidence(
            state=state,
            confidence=posterior,
            detection_score=detection_score,
            geometric_score=geometric_score,
            final_score=fused_score,
            should_track=should_track
        )
    
    def _bayesian_update(self, likelihood: float, prior: float) -> float:
        """
        Perform Bayesian update of belief.
        
        Args:
            likelihood: P(evidence | dynamic)
            prior: Prior P(static)
            
        Returns:
            Posterior probability P(dynamic | evidence)
        """
        # P(dynamic) = 1 - P(static)
        prior_dynamic = 1.0 - prior
        
        # P(evidence | static) - assume static objects rarely show motion
        likelihood_static = 0.1
        
        # P(evidence | dynamic) - dynamic objects show motion
        likelihood_dynamic = likelihood
        
        # Bayes rule: P(dynamic | evidence) = P(evidence | dynamic) * P(dynamic) / P(evidence)
        numerator = likelihood_dynamic * prior_dynamic
        denominator = (
            likelihood_dynamic * prior_dynamic +
            likelihood_static * prior
        )
        
        if denominator > 0:
            posterior = numerator / denominator
        else:
            posterior = prior_dynamic
        
        return posterior
    
    def _compute_temporal_consistency(self) -> float:
        """
        Compute temporal consistency from history.
        
        Returns:
            Temporal consistency score (0-1)
        """
        if len(self.history) == 0:
            return 0.5
        
        # Compute moving average with recent bias
        weights = np.exp(np.linspace(-2, 0, len(self.history)))
        weights = weights / weights.sum()
        
        weighted_score = np.sum(np.array(self.history) * weights)
        return float(weighted_score)
    
    def _update_history(self, score: float):
        """Update temporal history with new score."""
        self.history.append(score)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def _classify_state(self, confidence: float) -> ObjectState:
        """
        Classify object state based on confidence.
        
        Args:
            confidence: Fused confidence score
            
        Returns:
            ObjectState classification
        """
        if confidence >= self.confidence_threshold:
            return ObjectState.DYNAMIC
        elif confidence <= self.uncertainty_threshold:
            return ObjectState.STATIC
        else:
            return ObjectState.UNCERTAIN
    
    def _should_track_object(self, state: ObjectState, confidence: float) -> bool:
        """
        Determine if object should be tracked.
        
        Args:
            state: Classified object state
            confidence: Confidence score
            
        Returns:
            True if object should be tracked
        """
        if state == ObjectState.STATIC:
            # Track static objects with high confidence for SLAM
            return True
        elif state == ObjectState.DYNAMIC:
            # Exclude dynamic objects from SLAM
            return False
        else:
            # For uncertain cases, use conservative approach
            return confidence < 0.5
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the fusion process.
        
        Returns:
            Dictionary with fusion statistics
        """
        if len(self.history) == 0:
            return {
                'mean_score': 0.0,
                'std_score': 0.0,
                'temporal_consistency': 0.0,
                'history_length': 0
            }
        
        history_array = np.array(self.history)
        return {
            'mean_score': float(np.mean(history_array)),
            'std_score': float(np.std(history_array)),
            'temporal_consistency': self._compute_temporal_consistency(),
            'history_length': len(self.history)
        }
    
    def reset(self):
        """Reset the fusion state."""
        self.history = []
