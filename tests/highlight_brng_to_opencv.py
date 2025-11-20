#!/usr/bin/env python3
"""
Script to analyze pixels outside broadcast range using ffmpeg and OpenCV.
Processes entire video to highlight out-of-range pixels, then analyzes their distribution.
"""

import subprocess
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import argparse
from collections import defaultdict
import json

class BroadcastRangeAnalyzer:
    def __init__(self, input_video, output_dir="output", temp_dir="temp"):
        self.input_video = Path(input_video)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Define output paths
        self.processed_video = self.temp_dir / f"{self.input_video.stem}_highlighted.mp4"
        self.analysis_results = self.output_dir / f"{self.input_video.stem}_analysis.json"
        self.heatmap_video = self.output_dir / f"{self.input_video.stem}_heatmap.mp4"
        
    def process_with_ffmpeg(self, scale="720x486"):
        """
        Process video with ffmpeg to highlight out-of-broadcast-range pixels.
        """
        print(f"Processing {self.input_video} with ffmpeg...")
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(self.input_video),
            "-vf", f"signalstats=out=brng:color=cyan,scale={scale}",
            "-c:v", "libx264",  # Ensure good quality encoding
            "-preset", "fast",
            "-y",  # Overwrite output file
            str(self.processed_video)
        ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, 
                                 capture_output=True, 
                                 text=True, 
                                 check=True)
            print(f"FFmpeg processing completed. Output: {self.processed_video}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            return False
            
    def detect_cyan_pixels(self, frame):
        """
        Detect cyan colored pixels (highlights from ffmpeg signalstats).
        Returns binary mask of cyan pixels.
        """
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for cyan color in HSV
        # Cyan is around 90-100 in OpenCV HSV (H range 0-179)
        lower_cyan = np.array([85, 50, 50])
        upper_cyan = np.array([105, 255, 255])
        
        # Create mask for cyan pixels
        mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        return mask
    
    def find_pixel_clusters(self, mask):
        """
        Find and analyze clusters of highlighted pixels.
        Returns list of contour information.
        """
        # Find contours of cyan pixel clusters
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clusters = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:  # Filter out very small noise
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center point
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                clusters.append({
                    'area': area,
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'contour': contour
                })
        
        return clusters
    
    def analyze_video(self):
        """
        Analyze the processed video to find patterns in out-of-range pixels.
        """
        print(f"Analyzing processed video: {self.processed_video}")
        
        cap = cv2.VideoCapture(str(self.processed_video))
        if not cap.isOpened():
            print("Error: Could not open processed video")
            return None
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Initialize analysis data
        frame_analysis = []
        pixel_heatmap = np.zeros((height, width), dtype=np.float32)
        region_frequency = defaultdict(int)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect cyan pixels
            cyan_mask = self.detect_cyan_pixels(frame)
            
            # Find clusters
            clusters = self.find_pixel_clusters(cyan_mask)
            
            # Update heatmap
            pixel_heatmap += cyan_mask.astype(np.float32) / 255.0
            
            # Analyze frame
            total_cyan_pixels = np.sum(cyan_mask > 0)
            frame_data = {
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'total_cyan_pixels': int(total_cyan_pixels),
                'num_clusters': len(clusters),
                'clusters': []
            }
            
            # Process clusters
            for cluster in clusters:
                x, y, w, h = cluster['bbox']
                cx, cy = cluster['center']
                
                # Define regions (you can adjust these divisions)
                region_x = "left" if cx < width/3 else "center" if cx < 2*width/3 else "right"
                region_y = "top" if cy < height/3 else "middle" if cy < 2*height/3 else "bottom"
                region = f"{region_y}_{region_x}"
                region_frequency[region] += 1
                
                cluster_data = {
                    'area': cluster['area'],
                    'bbox': [x, y, w, h],
                    'center': [cx, cy],
                    'region': region
                }
                frame_data['clusters'].append(cluster_data)
            
            frame_analysis.append(frame_data)
            frame_count += 1
            
            # Progress indicator
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
        # Compile final analysis
        analysis = {
            'video_info': {
                'source': str(self.input_video),
                'processed': str(self.processed_video),
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': total_frames / fps
            },
            'summary': {
                'total_frames_with_issues': sum(1 for f in frame_analysis if f['total_cyan_pixels'] > 0),
                'average_pixels_per_frame': np.mean([f['total_cyan_pixels'] for f in frame_analysis]),
                'max_pixels_in_frame': max([f['total_cyan_pixels'] for f in frame_analysis]),
                'region_frequency': dict(region_frequency)
            },
            'frame_analysis': frame_analysis
        }
        
        # Save analysis results
        with open(self.analysis_results, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Analysis complete. Results saved to: {self.analysis_results}")
        
        # Create heatmap video
        self.create_heatmap_video(pixel_heatmap, fps)
        
        return analysis
    
    def create_heatmap_video(self, heatmap, fps):
        """
        Create a video showing the heatmap of problematic pixels.
        """
        print("Creating heatmap video...")
        
        # Normalize heatmap for visualization
        normalized_heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        
        # Apply colormap
        colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.heatmap_video), fourcc, fps, 
                             (colored_heatmap.shape[1], colored_heatmap.shape[0]))
        
        # Write the same heatmap frame for the duration of the video
        cap = cv2.VideoCapture(str(self.processed_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        for _ in range(total_frames):
            out.write(colored_heatmap)
        
        out.release()
        print(f"Heatmap video saved to: {self.heatmap_video}")
    
    def print_summary(self, analysis):
        """
        Print a summary of the analysis results.
        """
        summary = analysis['summary']
        video_info = analysis['video_info']
        
        print("\n" + "="*60)
        print("BROADCAST RANGE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Source Video: {video_info['source']}")
        print(f"Duration: {video_info['duration']:.2f} seconds")
        print(f"Total Frames: {video_info['total_frames']}")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        print()
        print("ISSUE SUMMARY:")
        print(f"Frames with out-of-range pixels: {summary['total_frames_with_issues']}")
        print(f"Percentage of problematic frames: {summary['total_frames_with_issues']/video_info['total_frames']*100:.2f}%")
        print(f"Average problematic pixels per frame: {summary['average_pixels_per_frame']:.1f}")
        print(f"Maximum problematic pixels in a single frame: {summary['max_pixels_in_frame']}")
        print()
        print("REGIONAL DISTRIBUTION:")
        for region, count in sorted(summary['region_frequency'].items()):
            print(f"  {region}: {count} clusters")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze broadcast range issues in video')
    parser.add_argument('input_video', help='Path to input video file (mkv, mp4, etc.)')
    parser.add_argument('--output-dir', default='output', help='Output directory for results')
    parser.add_argument('--temp-dir', default='temp', help='Temporary directory for processing')
    parser.add_argument('--scale', default='720x486', help='Scale for ffmpeg processing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file '{args.input_video}' not found")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = BroadcastRangeAnalyzer(args.input_video, args.output_dir, args.temp_dir)
    
    # Process with ffmpeg
    if not analyzer.process_with_ffmpeg(args.scale):
        print("FFmpeg processing failed")
        sys.exit(1)
    
    # Analyze the processed video
    analysis = analyzer.analyze_video()
    if analysis is None:
        print("Video analysis failed")
        sys.exit(1)
    
    # Print summary
    analyzer.print_summary(analysis)
    
    print(f"\nFiles generated:")
    print(f"- Analysis results: {analyzer.analysis_results}")
    print(f"- Heatmap video: {analyzer.heatmap_video}")
    print(f"- Processed video: {analyzer.processed_video}")

if __name__ == "__main__":
    main()
