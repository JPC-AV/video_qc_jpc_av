#!/usr/bin/env python3
"""
FFprobe Signalstats Analyzer

Uses ffprobe with lavfi to analyze broadcast range violations (BRNG) in video files.
Can work with border detection data from border_detector.py or analyze full frames.

This script focuses specifically on FFprobe signalstats analysis for broadcast compliance.
"""

import json
import subprocess
import shlex
import numpy as np
from pathlib import Path
import cv2


class FFprobeAnalyzer:
    """
    Analyzes video files using FFprobe signalstats for broadcast range compliance
    """
    
    def __init__(self, video_path):
        self.video_path = str(video_path)
        
        # Check FFprobe availability
        self.check_ffprobe()
        
        # Get video properties
        self.get_video_properties()
        
        # Detect bit depth and set appropriate thresholds
        self.bit_depth = self.detect_bit_depth()
        self.set_broadcast_thresholds()
        
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
            raise
        except Exception as e:
            print(f"⚠️  Could not check FFprobe: {e}")
            
    def get_video_properties(self):
        """Get basic video properties using FFprobe"""
        try:
            # Get both stream and format information
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,duration,r_frame_rate:format=duration',
                '-of', 'json',
                self.video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise ValueError(f"Could not get video properties: {result.stderr}")
                
            data = json.loads(result.stdout)
            stream = data['streams'][0]
            format_data = data.get('format', {})
            
            self.width = int(stream['width'])
            self.height = int(stream['height'])
            
            # Try to get duration from stream first, then format, then calculate from frames
            self.duration = None
            if 'duration' in stream:
                self.duration = float(stream['duration'])
            elif 'duration' in format_data:
                self.duration = float(format_data['duration'])
            
            # Parse frame rate
            fps_str = stream.get('r_frame_rate', '25/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                self.fps = num / den if den > 0 else 25
            else:
                self.fps = float(fps_str)
            
            # If duration is still None, try to calculate from frame count
            if self.duration is None or self.duration == 0:
                try:
                    # Get frame count using a different ffprobe command
                    count_cmd = [
                        'ffprobe', '-v', 'error',
                        '-select_streams', 'v:0',
                        '-count_packets',
                        '-show_entries', 'stream=nb_read_packets',
                        '-of', 'csv=p=0',
                        self.video_path
                    ]
                    count_result = subprocess.run(count_cmd, capture_output=True, text=True)
                    if count_result.returncode == 0:
                        frame_count = int(count_result.stdout.strip())
                        self.duration = frame_count / self.fps if self.fps > 0 else 0
                        print(f"✓ Calculated duration from {frame_count} frames")
                except:
                    # Final fallback - use a very conservative estimate
                    self.duration = 0
                    
            print(f"✓ Video properties: {self.width}x{self.height}, {self.fps:.2f}fps, {self.duration:.1f}s")
            
        except Exception as e:
            print(f"Error getting video properties: {e}")
            # Fallback to OpenCV for basic properties
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = total_frames / self.fps if self.fps > 0 else 0
                cap.release()
                print(f"✓ Using OpenCV fallback: {self.width}x{self.height}, {self.fps:.2f}fps, {self.duration:.1f}s")
            else:
                raise ValueError(f"Cannot open video file: {self.video_path}")
    
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
            return self.parse_ffprobe_output(result.stdout, region_name)
            
        except Exception as e:
            print(f"Error running FFprobe: {e}")
            return None
            
    def parse_ffprobe_output(self, output, region_name="frame"):
        """
        Parse FFprobe CSV output for BRNG values
        """
        results = {
            'region_name': region_name,
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
        
    def analyze_detailed_stats(self, active_area=None, start_time=120, duration=10, region_name="frame"):
        """
        Get more detailed statistics including YMIN, YMAX if available
        """
        # Limit detailed analysis duration for performance
        duration = min(duration, 10)
        
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
                'region_name': region_name,
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
    
    def analyze_border_regions_from_data(self, border_regions, start_time=120, duration=60):
        """
        Analyze border regions using pre-calculated border data
        
        Args:
            border_regions: Dictionary of border regions from border detector
            start_time: Start time for analysis
            duration: Duration of analysis
        """
        results = {}
        
        if not border_regions:
            print("No border regions provided")
            return results
        
        for region_name, coords in border_regions.items():
            if coords:
                x, y, w, h = coords
                print(f"\nAnalyzing {region_name}: {w}x{h} at ({x},{y})")
                results[region_name] = self.analyze_with_ffprobe(
                    active_area=coords,
                    start_time=start_time,
                    duration=duration,
                    region_name=region_name
                )
        
        return results
    
    def load_border_data(self, border_data_path):
        """
        Load border detection data from JSON file created by border_detector.py
        """
        try:
            with open(border_data_path, 'r') as f:
                border_data = json.load(f)
            
            print(f"✓ Loaded border data from {border_data_path}")
            return border_data
        
        except Exception as e:
            print(f"⚠️ Could not load border data: {e}")
            return None


def analyze_video_signalstats(video_path, border_data_path=None, output_dir=None, 
                             start_time=120, duration=60):
    """
    Main function to analyze video signalstats with optional border data
    
    Args:
        video_path: Path to video file
        border_data_path: Optional path to border data JSON from border_detector.py
        output_dir: Output directory for results
        start_time: Start analysis at this time in seconds
        duration: Duration of analysis in seconds
    
    Returns:
        Dictionary with analysis results
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
    
    analyzer = FFprobeAnalyzer(video_path)
    
    # Load border data if provided
    border_data = None
    active_area = None
    
    if border_data_path:
        border_data = analyzer.load_border_data(border_data_path)
        if border_data and border_data.get('active_area'):
            active_area = tuple(border_data['active_area'])
            print(f"Using active area from border data: {active_area}")
    
    # Analyze full frame
    print("\nAnalyzing full frame with FFprobe signalstats...")
    full_results = analyzer.analyze_with_ffprobe(
        active_area=None, 
        start_time=start_time,
        duration=duration,
        region_name="full frame"
    )
    
    # Analyze active area if available
    active_results = None
    if active_area:
        print("\nAnalyzing active area only...")
        active_results = analyzer.analyze_with_ffprobe(
            active_area=active_area,
            start_time=start_time,
            duration=duration,
            region_name="active area"
        )
    
    # Analyze border regions if available
    border_results = None
    if border_data and border_data.get('border_regions'):
        print("\nAnalyzing border regions...")
        border_results = analyzer.analyze_border_regions_from_data(
            border_data['border_regions'],
            start_time=start_time,
            duration=duration
        )
    
    # Get detailed stats for shorter duration
    detail_duration = min(duration, 10)
    print(f"\nGetting detailed statistics ({detail_duration}s sample)...")
    
    full_detailed = analyzer.analyze_detailed_stats(
        active_area=None,
        start_time=start_time,
        duration=detail_duration,
        region_name="full frame detailed"
    )
    
    active_detailed = None
    if active_area:
        active_detailed = analyzer.analyze_detailed_stats(
            active_area=active_area,
            start_time=start_time,
            duration=detail_duration,
            region_name="active area detailed"
        )
    
    # Create comprehensive report
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
        'border_data_used': border_data_path is not None,
        'active_area': list(active_area) if active_area else None,
        'results': {
            'full_frame': full_results,
            'active_area': active_results,
            'border_regions': border_results,
            'detailed_stats': {
                'full_frame': full_detailed,
                'active_area': active_detailed
            }
        }
    }
    
    # Analysis and diagnosis
    print("\n" + "="*80)
    print("FFPROBE SIGNALSTATS ANALYSIS RESULTS")
    print("="*80)
    print(f"\nVideo: {analyzer.bit_depth}-bit, using Y thresholds {analyzer.ymin_threshold}-{analyzer.ymax_threshold}")
    print(f"Analysis period: {start_time//60}:{start_time%60:02d} to {(start_time+duration)//60}:{(start_time+duration)%60:02d}")
    
    if full_results:
        print(f"\n📺 FULL FRAME ANALYSIS:")
        print(f"   Frames with violations: {full_results['frames_with_violations']}/{full_results['frames_analyzed']} ({full_results['violation_percentage']:.1f}%)")
        print(f"   Average BRNG: {full_results['avg_brng']:.2f}%")
        print(f"   Maximum BRNG: {full_results['max_brng']:.2f}%")
    
    if active_results:
        print(f"\n🎯 ACTIVE AREA ANALYSIS:")
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
    elif full_results:
        if full_results['violation_percentage'] > 5:
            report['diagnosis'] = "⚠️ Significant BRNG violations found"
            print(f"\n⚠️ DIAGNOSIS: Video contains significant broadcast range violations")
        elif full_results['violation_percentage'] > 1:
            report['diagnosis'] = "⚠️ Minor BRNG violations found"
            print(f"\n⚠️ DIAGNOSIS: Video contains minor broadcast range violations")
        else:
            report['diagnosis'] = "✓ Video appears broadcast-compliant"
            print(f"\n✅ DIAGNOSIS: Video appears to be within broadcast range")
    
    # Save JSON report
    json_path = output_dir / f"{video_path.stem}_signalstats_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed analysis report saved: {json_path}")
    
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        border_data_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        video_file = "JPC_AV_00011.mkv"
        border_data_file = None
    
    # Process with FFprobe signalstats analysis
    results = analyze_video_signalstats(
        video_file, 
        border_data_path=border_data_file,
        start_time=120,  # Start at 2 minutes
        duration=60      # Analyze for 1 minute
    )