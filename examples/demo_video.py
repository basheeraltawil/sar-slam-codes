"""
Demo: SAR-SLAM on Video Sequences

This example demonstrates SAR-SLAM processing on video input,
showing how the system handles dynamic scenes.
"""

import cv2
import numpy as np
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sar_slam import SARSLAM


def process_video(video_path, output_path=None, display=True):
    """
    Process video through SAR-SLAM pipeline.
    
    Args:
        video_path: Path to input video file or camera index
        output_path: Optional path to save output video
        display: Whether to display results in real-time
    """
    # Open video source
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps} fps")
    
    # Initialize video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize SAR-SLAM
    slam = SARSLAM(
        flow_threshold=2.0,
        ransac_threshold=1.0,
        detection_weight=0.4,
        geometric_weight=0.6,
        enable_visualization=True
    )
    
    frame_count = 0
    
    print("Processing video... (Press 'q' to quit)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        result = slam.process_frame(frame)
        
        # Create visualization
        vis_frame = slam.visualize(frame, result)
        
        # Save to output video
        if writer:
            writer.write(vis_frame)
        
        # Display
        if display:
            cv2.imshow('SAR-SLAM', vis_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Print progress
        if frame_count % 30 == 0:
            stats = slam.get_statistics()
            print(f"Frame {frame_count}: "
                  f"State={result.fused_evidence.state.name}, "
                  f"Quality={result.tracking_quality:.2f}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n=== Final Statistics ===")
    stats = slam.get_statistics()
    print(f"Total frames: {stats['total_frames']}")
    print(f"Static frames: {stats['static_frames']}")
    print(f"Dynamic frames: {stats['dynamic_frames']}")
    print(f"Uncertain frames: {stats['uncertain_frames']}")
    print(f"Fusion temporal consistency: {stats['fusion']['temporal_consistency']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='SAR-SLAM Video Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'video',
        help='Path to video file or camera index (0 for webcam)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Path to save output video',
        default=None
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable real-time display'
    )
    
    args = parser.parse_args()
    
    process_video(
        args.video,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == '__main__':
    main()
