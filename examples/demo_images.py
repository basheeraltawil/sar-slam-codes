"""
Demo: SAR-SLAM on Image Sequences

This example demonstrates SAR-SLAM processing on a sequence of images.
"""

import cv2
import numpy as np
import argparse
import sys
import os
import glob

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sar_slam import SARSLAM


def process_image_sequence(image_dir, output_dir=None, display=True):
    """
    Process image sequence through SAR-SLAM pipeline.
    
    Args:
        image_dir: Directory containing sequential images
        output_dir: Optional directory to save output images
        display: Whether to display results
    """
    # Get sorted list of images
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(image_dir, pattern)))
    
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"Error: No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SAR-SLAM
    slam = SARSLAM(
        flow_threshold=2.0,
        ransac_threshold=1.0,
        detection_weight=0.4,
        geometric_weight=0.6,
        enable_visualization=True
    )
    
    print("Processing images... (Press any key to continue, 'q' to quit)")
    
    for idx, image_path in enumerate(image_files):
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read {image_path}")
            continue
        
        # Process frame
        result = slam.process_frame(frame)
        
        # Create visualization
        vis_frame = slam.visualize(frame, result)
        
        # Save output
        if output_dir:
            output_path = os.path.join(
                output_dir,
                f"result_{idx:04d}.png"
            )
            cv2.imwrite(output_path, vis_frame)
        
        # Display
        if display:
            cv2.imshow('SAR-SLAM', vis_frame)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        
        # Print progress
        print(f"Image {idx+1}/{len(image_files)}: "
              f"State={result.fused_evidence.state.name}, "
              f"Quality={result.tracking_quality:.2f}")
    
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n=== Final Statistics ===")
    stats = slam.get_statistics()
    print(f"Total frames: {stats['total_frames']}")
    print(f"Static frames: {stats['static_frames']}")
    print(f"Dynamic frames: {stats['dynamic_frames']}")
    print(f"Uncertain frames: {stats['uncertain_frames']}")


def main():
    parser = argparse.ArgumentParser(
        description='SAR-SLAM Image Sequence Demo'
    )
    parser.add_argument(
        'image_dir',
        help='Directory containing sequential images'
    )
    parser.add_argument(
        '--output', '-o',
        help='Directory to save output images',
        default=None
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable interactive display'
    )
    
    args = parser.parse_args()
    
    process_image_sequence(
        args.image_dir,
        output_dir=args.output,
        display=not args.no_display
    )


if __name__ == '__main__':
    main()
