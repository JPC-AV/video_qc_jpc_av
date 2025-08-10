#!/usr/bin/env python3
"""
Active Area BRNG Analyzer

Analyzes broadcast range violations specifically in the active picture area,
using border detection data to exclude blanking/border regions.
Exports thumbnails of worst violation frames with timecode in filename.
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
        
        # Create thumbnails directory
        self.thumbnails_dir = self.output_dir / "brng_thumbnails"
        self.thumbnails_dir.mkdir(exist_ok=True)
        
        self.highlighted_video = self.temp_dir / f"{self.video_path.stem}_highlighted.mp4"
        self.analysis_output = self.output_dir / f"{self.video_path.stem}_active_brng_analysis.json"
        
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
        # Expanded range for better cyan detection
        lower_cyan = np.array([80, 40, 40])   # Slightly wider range
        upper_cyan = np.array([110, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Optional: Apply morphological operations to reduce noise
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def format_timecode(self, seconds):
        """Convert seconds to timecode format HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def format_filename_timecode(self, seconds):
        """Convert seconds to timecode format HH-MM-SS for filename"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * 30)  # Assuming 30fps for frame number
        return f"{hours:02d}-{minutes:02d}-{secs:02d}-{frames:02d}"
    
    def save_worst_frame_thumbnails(self, worst_frames, cap, num_thumbnails=5):
        """
        Save thumbnails of the worst frames with cyan highlights
        
        Args:
            worst_frames: List of worst frame data
            cap: Video capture object
            num_thumbnails: Number of thumbnails to save
        """
        print(f"\nSaving thumbnails of worst {num_thumbnails} frames...")
        
        saved_thumbnails = []
        
        for i, frame_data in enumerate(worst_frames[:num_thumbnails]):
            frame_idx = frame_data['frame']
            timestamp = frame_data['timestamp']
            violation_pct = frame_data['violation_percentage']
            
            # Seek to the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Generate filename with timecode
                timecode_str = self.format_filename_timecode(timestamp)
                thumbnail_filename = f"{self.video_path.stem}_brng_frame_{timecode_str}_violation_{violation_pct:.4f}pct.jpg"
                thumbnail_path = self.thumbnails_dir / thumbnail_filename
                
                # Add text overlay with violation info
                h, w = frame.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Create a semi-transparent overlay for text background
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Add text
                text1 = f"Timecode: {self.format_timecode(timestamp)}"
                text2 = f"BRNG Violation: {violation_pct:.4f}% pixels"
                
                cv2.putText(frame, text1, (10, 25),
                           font, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, text2, (10, 50),
                           font, 0.7, (0, 255, 255), 2)
                
                # Save thumbnail
                cv2.imwrite(str(thumbnail_path), frame)
                
                saved_thumbnails.append({
                    'filename': thumbnail_filename,
                    'path': str(thumbnail_path),
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'timecode': self.format_timecode(timestamp),
                    'violation_percentage': violation_pct
                })
                
                print(f"  ✓ Saved: {thumbnail_filename}")
            else:
                print(f"  ⚠️ Could not read frame {frame_idx}")
        
        return saved_thumbnails
    
    def analyze_highlighted_video(self, sample_every_n_frames=30):
        """
        Analyze the highlighted video to find BRNG violation patterns and save thumbnails.
        
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
                    
                    # Store frame data with timecode
                    timestamp = frame_idx / fps
                    frame_violations.append({
                        'frame': frame_idx,
                        'timestamp': timestamp,
                        'timecode': self.format_timecode(timestamp),
                        'violation_pixels': int(violation_pixels),
                        'violation_percentage': (violation_pixels / (width * height)) * 100
                    })
                
                analyzed_frames += 1
                
                # Progress indicator
                if analyzed_frames % 100 == 0:
                    print(f"  Analyzed {analyzed_frames} samples ({frame_idx}/{total_frames} frames)")
            
            frame_idx += 1
        
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
        
        # Save thumbnails of worst frames
        saved_thumbnails = []
        if worst_frames:
            # Reset video capture to beginning for thumbnail extraction
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            saved_thumbnails = self.save_worst_frame_thumbnails(worst_frames, cap, num_thumbnails=5)
        
        cap.release()
        
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
            'worst_frames': worst_frames,
            'saved_thumbnails': saved_thumbnails
        }
        
        # Save analysis
        with open(self.analysis_output, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✓ Analysis complete. Results saved to: {self.analysis_output}")
        if saved_thumbnails:
            print(f"✓ Saved {len(saved_thumbnails)} thumbnail(s) to: {self.thumbnails_dir}")
        
        return analysis
    
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
                print(f"  {i}. Frame {frame['frame']} ({frame['timecode']}) - {frame['violation_percentage']:.4f}% pixels")
        
        if analysis.get('saved_thumbnails'):
            print("\nSAVED THUMBNAILS:")
            for thumb in analysis['saved_thumbnails']:
                print(f"  - {thumb['filename']}")
        
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
    
    def cleanup(self, keep_highlighted=False):
        """
        Clean up temporary files
        
        Args:
            keep_highlighted: If True, keep the highlighted video for debugging
        """
        try:
            if not keep_highlighted and self.highlighted_video.exists():
                self.highlighted_video.unlink()
                print("✓ Removed highlighted video")
            elif keep_highlighted and self.highlighted_video.exists():
                print(f"ℹ️ Kept highlighted video: {self.highlighted_video}")
            
            # Remove temp directory if empty
            if self.temp_dir.exists():
                if not any(self.temp_dir.iterdir()):
                    self.temp_dir.rmdir()
                    print("✓ Cleaned up temporary directory")
                else:
                    print(f"ℹ️ Temp directory not empty: {self.temp_dir}")
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
    
    # Analyze the highlighted video and save thumbnails
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