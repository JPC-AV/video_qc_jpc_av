#!/usr/bin/env python3
"""
FFprobe Signalstats BRNG Analyzer with Border Detection

Uses ffprobe with lavfi to get BRNG statistics, which works more reliably
than ffmpeg for extracting signalstats data.

Modified to:
- Analyze the 3rd minute of video (2:00-3:00) 
- Handle 10-bit video properly
- Test border areas separately
- Use tighter active area detection
"""

import cv2
import numpy as np
import json
import subprocess
import shlex
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FFprobeSignalstatsAnalyzer:
    """
    Combines OpenCV border detection with FFprobe signalstats BRNG analysis
    """
    
    def __init__(self, video_path, sample_frames=30):
        self.video_path = str(video_path)
        self.sample_frames = sample_frames
        
        # Check FFprobe availability
        self.check_ffprobe()
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Detect bit depth and set appropriate thresholds
        self.bit_depth = self.detect_bit_depth()
        self.set_broadcast_thresholds()
        
    def detect_bit_depth(self):
        """Detect if video is 8-bit or 10-bit"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=pix_fmt',
                '-of', 'csv=p=0',
                self.video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            pix_fmt = result.stdout.strip()
            
            # Check for 10-bit formats
            if any(fmt in pix_fmt for fmt in ['p010', 'yuv420p10', 'yuv422p10', 'yuv444p10']):
                print(f"✓ Detected 10-bit video format: {pix_fmt}")
                return 10
            else:
                print(f"✓ Detected 8-bit video format: {pix_fmt}")
                return 8
                
        except Exception as e:
            print(f"⚠️  Could not detect bit depth: {e}, assuming 8-bit")
            return 8
    
    def set_broadcast_thresholds(self):
        """Set broadcast range thresholds based on bit depth"""
        if self.bit_depth == 10:
            self.ymin_threshold = 64   # 16 * 4 for 10-bit
            self.ymax_threshold = 940  # 235 * 4 for 10-bit
        else:
            self.ymin_threshold = 16
            self.ymax_threshold = 235
            
        print(f"✓ Using {self.bit_depth}-bit thresholds: Y={self.ymin_threshold}-{self.ymax_threshold}")
        
    def check_ffprobe(self):
        """Check if FFprobe is available"""
        try:
            result = subprocess.run(['ffprobe', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ FFprobe found")
            else:
                print("⚠️  FFprobe may not be properly installed")
        except FileNotFoundError:
            print("⚠️  FFprobe not found in PATH")
            print("   Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
        except Exception as e:
            print(f"⚠️  Could not check FFprobe: {e}")
            
    def detect_blanking_borders(self, threshold=10, edge_sample_width=100):
        """
        Detect borders with improved accuracy, especially for right side
        """
        frame_indices = np.linspace(0, self.total_frames - 1, 
                                   min(self.sample_frames, self.total_frames), 
                                   dtype=int)
        
        left_borders = []
        right_borders = []
        top_borders = []
        bottom_borders = []
        
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Detect left border - scan from left
            left = 0
            for x in range(min(edge_sample_width, w)):
                # Use mean instead of max for more robust detection
                if np.mean(gray[:, x]) > threshold:
                    left = x
                    break
                    
            # Detect right border - scan from right, more thorough
            right = w
            for x in range(w-1, max(w-edge_sample_width-1, -1), -1):
                if np.mean(gray[:, x]) > threshold:
                    right = x + 1
                    break
                    
            # For top/bottom, focus on center area to avoid side borders
            center_start = w // 3
            center_end = 2 * w // 3
            
            # Detect top border
            top = 0
            for y in range(min(20, h)):
                if np.mean(gray[y, center_start:center_end]) > threshold:
                    top = y
                    break
                    
            # Detect bottom border
            bottom = h
            for y in range(h-1, max(h-20, -1), -1):
                if np.mean(gray[y, center_start:center_end]) > threshold:
                    bottom = y + 1
                    break
                    
            left_borders.append(left)
            right_borders.append(right)
            top_borders.append(top)
            bottom_borders.append(bottom)
            
        if not left_borders:
            return None
            
        # Calculate stable borders with more conservative approach
        median_left = int(np.median(left_borders))
        median_right = int(np.median(right_borders))
        
        # Add some padding for tighter active area
        padding = 5
        median_left += padding
        median_right -= padding
        
        top_unique, top_counts = np.unique(top_borders, return_counts=True)
        mode_top = int(top_unique[np.argmax(top_counts)]) + padding
        
        bottom_unique, bottom_counts = np.unique(bottom_borders, return_counts=True)
        mode_bottom = int(bottom_unique[np.argmax(bottom_counts)]) - padding
        
        active_width = median_right - median_left
        active_height = mode_bottom - mode_top
        
        if active_width < 100 or active_height < 100:
            print("Warning: Detected active area seems too small")
            return None
            
        result = (median_left, mode_top, active_width, active_height)
        
        print(f"\nBorder detection statistics:")
        print(f"  Left border: median={median_left} (std={np.std(left_borders):.1f})")
        print(f"  Right border: median={median_right} (std={np.std(right_borders):.1f})")
        print(f"  Top border: mode={mode_top} (variations={len(top_unique)})")
        print(f"  Bottom border: mode={mode_bottom} (variations={len(bottom_unique)})")
        print(f"  Active area (with {padding}px padding): {active_width}x{active_height} at ({median_left},{mode_top})")
        
        return result
        
    def analyze_with_ffprobe(self, active_area=None, start_time=120, duration=60, region_name="frame"):
        """
        Use FFprobe with lavfi to analyze BRNG statistics
        
        Args:
            active_area: tuple (x, y, w, h) to crop to, or None for full frame
            start_time: start analysis at this time in seconds (default: 120 = 2 minutes)
            duration: duration of analysis in seconds (default: 60 = 1 minute)
            region_name: descriptive name for this region
        """
        # Check if video is long enough
        if self.duration < start_time + duration:
            print(f"⚠️  Video duration ({self.duration:.1f}s) is shorter than requested analysis period ({start_time}s + {duration}s)")
            if self.duration > start_time:
                duration = self.duration - start_time
                print(f"   Adjusting duration to {duration:.1f}s")
            else:
                print(f"   Falling back to first {min(duration, self.duration):.1f}s")
                start_time = 0
                duration = min(duration, self.duration)
        
        # Build filter chain
        filter_chain = f"movie={shlex.quote(self.video_path)}"
        
        # Add time selection - select frames between start_time and start_time+duration
        end_time = start_time + duration
        filter_chain += f",select='between(t\\,{start_time}\\,{end_time})'"
        
        # Add crop filter if active area is specified
        if active_area:
            x, y, w, h = active_area
            filter_chain += f",crop={w}:{h}:{x}:{y}"
            
        # Add signalstats
        filter_chain += ",signalstats=stat=brng"
        
        # Build FFprobe command
        cmd = [
            'ffprobe',
            '-f', 'lavfi',
            '-i', filter_chain,
            '-show_entries', 'frame_tags=lavfi.signalstats.BRNG',
            '-of', 'csv=p=0'
        ]
        
        print(f"\nRunning FFprobe signalstats analysis on {region_name}...")
        if active_area:
            x, y, w, h = active_area
            print(f"  Region: {w}x{h} at ({x},{y})")
        else:
            print(f"  Region: full frame {self.width}x{self.height}")
        print(f"  Time range: {start_time}s to {end_time}s ({duration}s duration)")
            
        # Run FFprobe and capture output
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFprobe error: {result.stderr}")
                return None
                
            # Parse CSV output
            return self.parse_ffprobe_output(result.stdout)
            
        except Exception as e:
            print(f"Error running FFprobe: {e}")
            return None
            
    def parse_ffprobe_output(self, output):
        """
        Parse FFprobe CSV output for BRNG values
        """
        results = {
            'frames_analyzed': 0,
            'frames_with_violations': 0,
            'brng_values': [],
            'avg_brng': 0.0,
            'max_brng': 0.0,
            'violation_percentage': 0.0
        }
        
        lines = output.strip().split('\n')
        
        for line in lines:
            if line.strip():
                try:
                    # CSV output should be just the BRNG value
                    brng_value = float(line.strip())
                    results['brng_values'].append(brng_value)
                    results['frames_analyzed'] += 1
                    
                    if brng_value > 0:
                        results['frames_with_violations'] += 1
                        
                except ValueError:
                    # Skip non-numeric lines
                    pass
                    
        # Calculate statistics
        if results['brng_values']:
            results['avg_brng'] = np.mean(results['brng_values'])
            results['max_brng'] = np.max(results['brng_values'])
            
        if results['frames_analyzed'] > 0:
            results['violation_percentage'] = (results['frames_with_violations'] / 
                                             results['frames_analyzed']) * 100
                                             
        print(f"  Analyzed {results['frames_analyzed']} frames")
        print(f"  Frames with violations: {results['frames_with_violations']} ({results['violation_percentage']:.1f}%)")
        print(f"  Avg BRNG: {results['avg_brng']:.2f}%, Max BRNG: {results['max_brng']:.2f}%")
        
        return results
        
    def analyze_with_ffprobe_detailed(self, active_area=None, start_time=120, duration=60, region_name="frame"):
        """
        Get more detailed statistics including YMIN, YMAX if available
        """
        # Check if video is long enough
        if self.duration < start_time + duration:
            if self.duration > start_time:
                duration = self.duration - start_time
            else:
                start_time = 0
                duration = min(duration, self.duration)
        
        # Build filter chain
        filter_chain = f"movie={shlex.quote(self.video_path)}"
        
        # Add time selection
        end_time = start_time + duration
        filter_chain += f",select='between(t\\,{start_time}\\,{end_time})'"
        
        if active_area:
            x, y, w, h = active_area
            filter_chain += f",crop={w}:{h}:{x}:{y}"
            
        filter_chain += ",signalstats"  # Get all stats, not just BRNG
        
        # Build FFprobe command for multiple fields
        cmd = [
            'ffprobe',
            '-f', 'lavfi',
            '-i', filter_chain,
            '-show_entries', 'frame_tags=lavfi.signalstats.YMIN,lavfi.signalstats.YMAX,lavfi.signalstats.BRNG',
            '-of', 'json'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return None
                
            # Parse JSON output for more detailed analysis
            data = json.loads(result.stdout)
            
            results = {
                'frames_analyzed': 0,
                'frames_with_violations': 0,
                'ymin_violations': 0,
                'ymax_violations': 0,
                'brng_violations': 0,
                'avg_ymin': 0.0,
                'avg_ymax': 0.0,
                'min_ymin': float('inf'),
                'max_ymax': 0.0,
                'violation_percentage': 0.0
            }
            
            ymin_values = []
            ymax_values = []
            
            for frame in data.get('frames', []):
                tags = frame.get('tags', {})
                results['frames_analyzed'] += 1
                
                has_violation = False
                
                # Check YMIN
                ymin = float(tags.get('lavfi.signalstats.YMIN', self.ymin_threshold))
                ymin_values.append(ymin)
                if ymin < self.ymin_threshold:
                    results['ymin_violations'] += 1
                    has_violation = True
                    
                # Check YMAX
                ymax = float(tags.get('lavfi.signalstats.YMAX', self.ymax_threshold))
                ymax_values.append(ymax)
                if ymax > self.ymax_threshold:
                    results['ymax_violations'] += 1
                    has_violation = True
                    
                # Check BRNG
                brng = float(tags.get('lavfi.signalstats.BRNG', 0))
                if brng > 0:
                    results['brng_violations'] += 1
                    has_violation = True
                    
                if has_violation:
                    results['frames_with_violations'] += 1
                    
            # Calculate statistics
            if ymin_values:
                results['avg_ymin'] = np.mean(ymin_values)
                results['min_ymin'] = np.min(ymin_values)
            if ymax_values:
                results['avg_ymax'] = np.mean(ymax_values)
                results['max_ymax'] = np.max(ymax_values)
                
            if results['frames_analyzed'] > 0:
                results['violation_percentage'] = (results['frames_with_violations'] / 
                                                 results['frames_analyzed']) * 100
                                                 
            print(f"  Detailed stats for {region_name}:")
            print(f"    YMIN: avg={results['avg_ymin']:.1f}, min={results['min_ymin']:.1f}, violations={results['ymin_violations']}")
            print(f"    YMAX: avg={results['avg_ymax']:.1f}, max={results['max_ymax']:.1f}, violations={results['ymax_violations']}")
            print(f"    BRNG violations: {results['brng_violations']}")
                                                 
            return results
            
        except Exception as e:
            print(f"Error getting detailed stats: {e}")
            return None
    
    def analyze_border_regions(self, active_area, start_time=120, duration=60):
        """
        Analyze just the border regions to see what's in them
        """
        if not active_area:
            print("No active area detected, cannot analyze borders")
            return None
            
        x, y, w, h = active_area
        results = {}
        
        # Left border
        if x > 10:
            left_border = (0, 0, x, self.height)
            results['left_border'] = self.analyze_with_ffprobe(
                active_area=left_border, 
                start_time=start_time, 
                duration=duration,
                region_name="left border"
            )
        
        # Right border  
        right_border_start = x + w
        if right_border_start < self.width - 10:
            right_border = (right_border_start, 0, self.width - right_border_start, self.height)
            results['right_border'] = self.analyze_with_ffprobe(
                active_area=right_border,
                start_time=start_time,
                duration=duration, 
                region_name="right border"
            )
        
        # Top border
        if y > 10:
            top_border = (0, 0, self.width, y)
            results['top_border'] = self.analyze_with_ffprobe(
                active_area=top_border,
                start_time=start_time,
                duration=duration,
                region_name="top border"
            )
        
        # Bottom border
        bottom_border_start = y + h
        if bottom_border_start < self.height - 10:
            bottom_border = (0, bottom_border_start, self.width, self.height - bottom_border_start)
            results['bottom_border'] = self.analyze_with_ffprobe(
                active_area=bottom_border,
                start_time=start_time,
                duration=duration,
                region_name="bottom border"
            )
            
        return results
        
    def generate_comparison_report(self, output_path, active_area=None):
        """
        Generate visual comparison showing full frame vs active area with border regions highlighted
        """
        # Get a representative frame from the analysis period (around 2:30)
        target_frame = int(150 * self.fps)  # 2.5 minutes
        if target_frame >= self.total_frames:
            target_frame = self.total_frames // 2
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = self.cap.read()
        
        if not ret:
            return False
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Full frame with regions marked
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax1.imshow(frame_rgb)
        ax1.set_title('Full Frame with Border Analysis Regions')
        ax1.axis('off')
        
        if active_area:
            x, y, w, h = active_area
            
            # Mark active area in green
            active_rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                          edgecolor='lime', facecolor='none', 
                                          label='Active Area (Tighter)')
            ax1.add_patch(active_rect)
            
            # Mark border regions in red
            if x > 10:  # Left border
                left_rect = patches.Rectangle((0, 0), x, self.height, linewidth=2,
                                            edgecolor='red', facecolor='red', alpha=0.3,
                                            label='Border Regions')
                ax1.add_patch(left_rect)
            
            if x + w < self.width - 10:  # Right border
                right_rect = patches.Rectangle((x + w, 0), self.width - (x + w), self.height, 
                                             linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
                ax1.add_patch(right_rect)
                
            if y > 10:  # Top border
                top_rect = patches.Rectangle((0, 0), self.width, y, linewidth=2,
                                           edgecolor='red', facecolor='red', alpha=0.3)
                ax1.add_patch(top_rect)
                
            if y + h < self.height - 10:  # Bottom border
                bottom_rect = patches.Rectangle((0, y + h), self.width, self.height - (y + h), 
                                              linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
                ax1.add_patch(bottom_rect)
            
            ax1.legend()
            
            # Active area only
            active_frame = frame_rgb[y:y+h, x:x+w]
            ax2.imshow(active_frame)
            ax2.set_title('Tighter Active Picture Area Only')
            ax2.axis('off')
            
            # Add text annotations
            border_info = f'Border sizes: L={x}px, R={self.width-x-w}px, T={y}px, B={self.height-y-h}px'
            fig.text(0.5, 0.02, border_info, ha='center', fontsize=10)
            fig.text(0.5, 0.95, f'{self.bit_depth}-bit video | Analysis period: 2:00-3:00', 
                    ha='center', fontsize=12, weight='bold')
        else:
            ax2.text(0.5, 0.5, 'No borders detected', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    def close(self):
        self.cap.release()


def process_video_with_ffprobe_analysis(video_path, output_dir=None, start_time=120, duration=60):
    """
    Main processing function using FFprobe signalstats for broadcast range analysis
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for reports
        start_time: Start analysis at this time in seconds (default: 120 = 2 minutes)
        duration: Duration of analysis in seconds (default: 60 = 1 minute)
    """
    video_path = Path(video_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = video_path.parent
        
    print(f"Processing: {video_path.name}")
    print(f"Using FFprobe signalstats BRNG analysis")
    print(f"Analyzing from {start_time//60}:{start_time%60:02d} to {(start_time+duration)//60}:{(start_time+duration)%60:02d}")
    
    analyzer = FFprobeSignalstatsAnalyzer(video_path)
    
    # Check video duration
    print(f"Video duration: {analyzer.duration:.1f} seconds ({analyzer.duration//60:.0f}:{analyzer.duration%60:02.0f})")
    
    # Step 1: Detect borders with improved algorithm
    print("\nDetecting active picture area (tighter detection)...")
    active_area = analyzer.detect_blanking_borders(threshold=10)
    
    # Step 2: Analyze full frame with FFprobe
    print("\nAnalyzing full frame with FFprobe signalstats...")
    full_results = analyzer.analyze_with_ffprobe(
        active_area=None, 
        start_time=start_time,
        duration=duration,
        region_name="full frame"
    )
    
    # Step 3: Analyze active area only (tighter)
    active_results = None
    if active_area:
        print("\nAnalyzing tighter active area only...")
        active_results = analyzer.analyze_with_ffprobe(
            active_area=active_area,
            start_time=start_time,
            duration=duration,
            region_name="tighter active area"
        )
    
    # Step 4: NEW - Analyze border regions separately
    border_results = None
    if active_area:
        print("\nAnalyzing border regions separately...")
        border_results = analyzer.analyze_border_regions(
            active_area=active_area,
            start_time=start_time,
            duration=duration
        )
    
    # Step 5: Get detailed stats (limit to 10 seconds for performance)
    print("\nGetting detailed statistics...")
    detail_duration = min(duration, 10)
    full_detailed = analyzer.analyze_with_ffprobe_detailed(
        active_area=None,
        start_time=start_time,
        duration=detail_duration,
        region_name="full frame detailed"
    )
    
    active_detailed = None
    if active_area:
        active_detailed = analyzer.analyze_with_ffprobe_detailed(
            active_area=active_area,
            start_time=start_time,
            duration=detail_duration,
            region_name="active area detailed"
        )
    
    # Step 6: Generate visual comparison
    comparison_path = output_dir / f"{video_path.stem}_ffprobe_comparison.jpg"
    analyzer.generate_comparison_report(comparison_path, active_area)
    print(f"\nVisual comparison saved: {comparison_path}")
    
    # Step 7: Create comprehensive report
    report = {
        'video_file': str(video_path),
        'bit_depth': analyzer.bit_depth,
        'broadcast_thresholds': {
            'ymin': analyzer.ymin_threshold,
            'ymax': analyzer.ymax_threshold
        },
        'analysis_method': 'FFprobe signalstats',
        'analysis_period': f'{start_time}s to {start_time + duration}s',
        'sample_duration': duration,
        'active_area': list(active_area) if active_area else None,
        'ffprobe_results': {
            'full_frame': full_results,
            'active_area_only': active_results,
            'border_regions': border_results,
            'full_frame_detailed': full_detailed,
            'active_area_detailed': active_detailed
        }
    }
    
    # Enhanced diagnosis
    print("\n" + "="*80)
    print("ENHANCED FFPROBE SIGNALSTATS ANALYSIS RESULTS")
    print("="*80)
    print(f"\nVideo: {analyzer.bit_depth}-bit, using Y thresholds {analyzer.ymin_threshold}-{analyzer.ymax_threshold}")
    print(f"Analysis period: {start_time//60}:{start_time%60:02d} to {(start_time+duration)//60}:{(start_time+duration)%60:02d}")
    
    if full_results:
        print(f"\n📺 FULL FRAME ANALYSIS:")
        print(f"   Frames with violations: {full_results['frames_with_violations']}/{full_results['frames_analyzed']} ({full_results['violation_percentage']:.1f}%)")
        print(f"   Average BRNG: {full_results['avg_brng']:.2f}%")
        print(f"   Maximum BRNG: {full_results['max_brng']:.2f}%")
    
    if active_results:
        print(f"\n🎯 TIGHTER ACTIVE AREA ANALYSIS:")
        print(f"   Frames with violations: {active_results['frames_with_violations']}/{active_results['frames_analyzed']} ({active_results['violation_percentage']:.1f}%)")
        print(f"   Average BRNG: {active_results['avg_brng']:.2f}%")
        print(f"   Maximum BRNG: {active_results['max_brng']:.2f}%")
    
    if border_results:
        print(f"\n🔴 BORDER REGIONS ANALYSIS:")
        for region, data in border_results.items():
            if data:
                print(f"   {region}: {data['violation_percentage']:.1f}% violations, avg BRNG: {data['avg_brng']:.2f}%")
    
    # Determine diagnosis
    if active_area and active_results and full_results:
        full_violations = full_results['violation_percentage']
        active_violations = active_results['violation_percentage']
        
        if full_violations > 5 and active_violations < 1:
            report['diagnosis'] = "✓ BRNG violations are primarily in blanking/border areas"
            print(f"\n✅ DIAGNOSIS: Violations reduced by {full_violations - active_violations:.1f}% when excluding borders")
            print("   → Video content appears broadcast-safe, violations are in inactive areas")
        elif active_violations > 5:
            report['diagnosis'] = "⚠️ Significant BRNG violations found in active picture content"
            print(f"\n⚠️ DIAGNOSIS: Active content contains significant broadcast range violations")
            print("   → Video may need levels adjustment for broadcast compliance")
        elif active_violations > 1:
            report['diagnosis'] = "⚠️ Minor BRNG violations found in active picture content"
            print(f"\n⚠️ DIAGNOSIS: Active content contains minor broadcast range violations")
            print("   → Review recommended for broadcast compliance")
        else:
            report['diagnosis'] = "✓ Video appears broadcast-compliant"
            print(f"\n✅ DIAGNOSIS: Video appears to be within broadcast range")
    
    # Save JSON report
    json_path = output_dir / f"{video_path.stem}_ffprobe_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed analysis report saved: {json_path}")
    
    analyzer.close()
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = "JPC_AV_00011.mkv"
    
    # Process with enhanced FFprobe signalstats - analyze 3rd minute (2:00 to 3:00)
    results = process_video_with_ffprobe_analysis(
        video_file, 
        start_time=300,  # Start at 2 minutes (120 seconds)
        duration=60      # Analyze for 1 minute (60 seconds)
    )