#!/usr/bin/env python3
"""
Active Area BRNG Analyzer

Analyzes broadcast range violations specifically in the active picture area,
using border detection data to exclude blanking/border regions.
"""

import subprocess
import cv2
import numpy as np
import json
from pathlib import Path
import sys
from collections import defaultdict


class ActiveAreaBrngAnalyzer:
    """
    Analyzes BRNG violations in the active picture area only
    """
    
    def __init__(self, video_path, border_data_path=None, output_dir=None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent
        self.output_dir.mkdir(exist_ok=True)
        
        # Load border data if provided
        self.active_area = None
        self.border_data = None
        if border_data_path:
            self.load_border_data(border_data_path)
        
        # Define output paths
        self.temp_dir = self.output_dir / "temp_brng"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.highlighted_video = self.temp_dir / f"{self.video_path.stem}_highlighted.mp4"
        self.analysis_output = self.output_dir / f"{self.video_path.stem}_active_brng_analysis.json"
        self.heatmap_image = self.output_dir / f"{self.video_path.stem}_brng_heatmap.png"
        
    def load_border_data(self, border_data_path):
        """Load border detection data"""
        try:
            with open(border_data_path, 'r') as f:
                self.border_data = json.load(f)
            
            if self.border_data and self.border_data.get('active_area'):
                self.active_area = tuple(self.border_data['active_area'])
                print(f"✓ Loaded border data. Active area: {self.active_area}")
            else:
                print("⚠️ Border data doesn't contain active area")
        except Exception as e:
            print(f"⚠️ Could not load border data: {e}")
    
    def process_with_ffmpeg(self, duration_limit=300):
        """
        Process video with ffmpeg signalstats, optionally cropping to active area.
        
        Args:
            duration_limit: Maximum duration in seconds to process (default 300 = 5 minutes)
        """
        print(f"\nProcessing video with ffmpeg signalstats...")
        
        # Build filter chain
        filters = []
        
        # Add crop filter if active area is known
        if self.active_area:
            x, y, w, h = self.active_area
            filters.append(f"crop={w}:{h}:{x}:{y}")
            print(f"  Cropping to active area: {w}x{h} at ({x},{y})")
        else:
            print("  No border data - analyzing full frame")
        
        # Add signalstats filter to highlight BRNG violations
        filters.append("signalstats=out=brng:color=cyan")
        
        # Join filters
        filter_chain = ",".join(filters)
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(self.video_path),
            "-t", str(duration_limit),  # Limit duration for faster processing
            "-vf", filter_chain,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            str(self.highlighted_video)
        ]
        
        print(f"  Processing up to {duration_limit} seconds...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ FFmpeg processing complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg error: {e}")
            if e.stderr:
                print(f"  Error details: {e.stderr[:500]}")
            return False
    
    def detect_cyan_pixels(self, frame):
        """Detect cyan-highlighted pixels from signalstats"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Cyan in HSV (OpenCV uses H: 0-179, S: 0-255, V: 0-255)
        lower_cyan = np.array([85, 50, 50])
        upper_cyan = np.array([105, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        return mask
    
    def analyze_highlighted_video(self, sample_every_n_frames=30):
        """
        Analyze the highlighted video to find BRNG violation patterns.
        
        Args:
            sample_every_n_frames: Sample every Nth frame for faster processing
        """
        print(f"\nAnalyzing highlighted video for BRNG violations...")
        
        cap = cv2.VideoCapture(str(self.highlighted_video))
        if not cap.isOpened():
            print("✗ Could not open highlighted video")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
        print(f"  Sampling every {sample_every_n_frames} frames")
        
        # Initialize analysis data
        heatmap = np.zeros((height, width), dtype=np.float32)
        frame_violations = []
        region_stats = defaultdict(lambda: {'count': 0, 'total_pixels': 0})
        
        frame_idx = 0
        analyzed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames for analysis
            if frame_idx % sample_every_n_frames == 0:
                # Detect cyan pixels
                cyan_mask = self.detect_cyan_pixels(frame)
                violation_pixels = np.sum(cyan_mask > 0)
                
                if violation_pixels > 0:
                    # Update heatmap
                    heatmap += cyan_mask.astype(np.float32) / 255.0
                    
                    # Analyze regions
                    # Divide into 9 regions (3x3 grid)
                    h_third = height // 3
                    w_third = width // 3
                    
                    for row in range(3):
                        for col in range(3):
                            y1 = row * h_third
                            y2 = (row + 1) * h_third if row < 2 else height
                            x1 = col * w_third
                            x2 = (col + 1) * w_third if col < 2 else width
                            
                            region_mask = cyan_mask[y1:y2, x1:x2]
                            region_pixels = np.sum(region_mask > 0)
                            
                            if region_pixels > 0:
                                region_name = f"{'top' if row == 0 else 'middle' if row == 1 else 'bottom'}_{'left' if col == 0 else 'center' if col == 1 else 'right'}"
                                region_stats[region_name]['count'] += 1
                                region_stats[region_name]['total_pixels'] += region_pixels
                    
                    # Store frame data
                    frame_violations.append({
                        'frame': frame_idx,
                        'timestamp': frame_idx / fps,
                        'violation_pixels': int(violation_pixels),
                        'violation_percentage': (violation_pixels / (width * height)) * 100
                    })
                
                analyzed_frames += 1
                
                # Progress indicator
                if analyzed_frames % 100 == 0:
                    print(f"  Analyzed {analyzed_frames} samples ({frame_idx}/{total_frames} frames)")
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate statistics
        total_samples = analyzed_frames
        samples_with_violations = len(frame_violations)
        
        if samples_with_violations > 0:
            avg_violation_pixels = np.mean([f['violation_pixels'] for f in frame_violations])
            max_violation_pixels = max([f['violation_pixels'] for f in frame_violations])
            avg_violation_percentage = np.mean([f['violation_percentage'] for f in frame_violations])
            max_violation_percentage = max([f['violation_percentage'] for f in frame_violations])
        else:
            avg_violation_pixels = 0
            max_violation_pixels = 0
            avg_violation_percentage = 0
            max_violation_percentage = 0
        
        # Find worst frames
        worst_frames = sorted(frame_violations, 
                            key=lambda x: x['violation_pixels'], 
                            reverse=True)[:10]
        
        # Compile analysis results
        analysis = {
            'video_info': {
                'source': str(self.video_path),
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': total_frames / fps,
                'active_area': list(self.active_area) if self.active_area else None
            },
            'analysis_settings': {
                'sample_every_n_frames': sample_every_n_frames,
                'total_samples': total_samples,
                'analyzed_region': 'active_area' if self.active_area else 'full_frame'
            },
            'summary': {
                'samples_with_violations': samples_with_violations,
                'violation_percentage': (samples_with_violations / total_samples * 100) if total_samples > 0 else 0,
                'avg_violation_pixels': avg_violation_pixels,
                'max_violation_pixels': max_violation_pixels,
                'avg_violation_percentage': avg_violation_percentage,
                'max_violation_percentage': max_violation_percentage
            },
            'region_analysis': {
                region: {
                    'occurrences': stats['count'],
                    'avg_pixels': stats['total_pixels'] / stats['count'] if stats['count'] > 0 else 0
                }
                for region, stats in region_stats.items()
            },
            'worst_frames': worst_frames
        }
        
        # Save analysis
        with open(self.analysis_output, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✓ Analysis complete. Results saved to: {self.analysis_output}")
        
        # Create heatmap visualization
        if samples_with_violations > 0:
            self.create_heatmap(heatmap, analysis)
        
        return analysis
    
    def create_heatmap(self, heatmap, analysis):
        """Create a heatmap visualization of BRNG violations"""
        print("\nCreating heatmap visualization...")
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            normalized = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
            
            # Apply colormap
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            
            # Add text overlay with statistics
            h, w = colored.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Add title
            cv2.putText(colored, "BRNG Violation Heatmap", (10, 30),
                       font, 1, (255, 255, 255), 2)
            
            # Add statistics
            stats_text = [
                f"Active Area: {analysis['video_info']['active_area']}" if analysis['video_info']['active_area'] else "Full Frame",
                f"Samples with violations: {analysis['summary']['samples_with_violations']}",
                f"Max violation: {analysis['summary']['max_violation_percentage']:.3f}% pixels"
            ]
            
            y_pos = 60
            for text in stats_text:
                cv2.putText(colored, text, (10, y_pos),
                           font, 0.5, (255, 255, 255), 1)
                y_pos += 25
            
            # Save heatmap
            cv2.imwrite(str(self.heatmap_image), colored)
            print(f"✓ Heatmap saved to: {self.heatmap_image}")
    
    def print_summary(self, analysis):
        """Print analysis summary"""
        print("\n" + "="*80)
        print("ACTIVE AREA BRNG ANALYSIS RESULTS")
        print("="*80)
        
        info = analysis['video_info']
        summary = analysis['summary']
        
        if info['active_area']:
            print(f"Active Area: {info['active_area'][2]}x{info['active_area'][3]} at ({info['active_area'][0]},{info['active_area'][1]})")
        else:
            print("Analyzed: Full frame (no border data)")
        
        print(f"Video: {info['width']}x{info['height']}, {info['duration']:.1f}s")
        print(f"Samples analyzed: {analysis['analysis_settings']['total_samples']}")
        
        print("\nVIOLATION SUMMARY:")
        print(f"  Samples with violations: {summary['samples_with_violations']} ({summary['violation_percentage']:.1f}%)")
        print(f"  Average violation: {summary['avg_violation_pixels']:.1f} pixels ({summary['avg_violation_percentage']:.4f}%)")
        print(f"  Maximum violation: {summary['max_violation_pixels']} pixels ({summary['max_violation_percentage']:.4f}%)")
        
        if analysis['region_analysis']:
            print("\nREGIONAL DISTRIBUTION:")
            for region, stats in sorted(analysis['region_analysis'].items()):
                print(f"  {region}: {stats['occurrences']} occurrences, avg {stats['avg_pixels']:.1f} pixels")
        
        if analysis['worst_frames']:
            print("\nWORST FRAMES:")
            for i, frame in enumerate(analysis['worst_frames'][:5], 1):
                print(f"  {i}. Frame {frame['frame']} ({frame['timestamp']:.1f}s): {frame['violation_percentage']:.4f}% pixels")
        
        # Diagnosis
        print("\nDIAGNOSIS:")
        if summary['samples_with_violations'] == 0:
            print("  ✓ No BRNG violations detected in active area")
        elif summary['max_violation_percentage'] < 0.01:
            print("  ✓ Negligible BRNG violations (< 0.01% pixels)")
        elif summary['max_violation_percentage'] < 0.1:
            print("  ℹ️ Minor BRNG violations detected")
            print("     → Likely acceptable for broadcast")
        elif summary['max_violation_percentage'] < 1.0:
            print("  ⚠️ Moderate BRNG violations detected")
            print("     → Review recommended")
        else:
            print("  ⚠️ Significant BRNG violations detected")
            print("     → Levels adjustment recommended")
        
        print("="*80)
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.highlighted_video.exists():
                self.highlighted_video.unlink()
            if self.temp_dir.exists():
                self.temp_dir.rmdir()
            print("✓ Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️ Could not clean up temp files: {e}")


def analyze_active_area_brng(video_path, border_data_path=None, output_dir=None, 
                            duration_limit=300, sample_rate=30):
    """
    Main function to analyze BRNG violations in active picture area.
    
    Args:
        video_path: Path to video file
        border_data_path: Path to border detection JSON
        output_dir: Output directory for results
        duration_limit: Maximum seconds to process
        sample_rate: Sample every N frames
    
    Returns:
        Analysis results dictionary
    """
    analyzer = ActiveAreaBrngAnalyzer(video_path, border_data_path, output_dir)
    
    # Process with ffmpeg
    if not analyzer.process_with_ffmpeg(duration_limit):
        print("FFmpeg processing failed")
        return None
    
    # Analyze the highlighted video
    analysis = analyzer.analyze_highlighted_video(sample_rate)
    
    if analysis:
        analyzer.print_summary(analysis)
    
    # Clean up temp files
    analyzer.cleanup()
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze BRNG violations in active picture area')
    parser.add_argument('video_file', help='Path to video file')
    parser.add_argument('--border-data', help='Path to border detection JSON file')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--duration', type=int, default=300, help='Max duration to analyze (seconds)')
    parser.add_argument('--sample-rate', type=int, default=30, help='Sample every N frames')
    
    args = parser.parse_args()
    
    results = analyze_active_area_brng(
        args.video_file,
        args.border_data,
        args.output_dir,
        args.duration,
        args.sample_rate
    )