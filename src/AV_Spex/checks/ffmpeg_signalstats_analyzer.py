#!/usr/bin/env python3
"""
FFprobe Signalstats Analyzer (Simplified)

Uses ffprobe with lavfi to analyze broadcast range violations (BRNG) in video files.
Focuses on active area and border regions when border detection data is available.
Skips full frame analysis to improve performance.
"""

import json
import subprocess
import shlex
import numpy as np
from pathlib import Path
import cv2

from AV_Spex.utils.log_setup import logger

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
        
        # Detect bit depth
        self.bit_depth = self.detect_bit_depth()
        
    def check_ffprobe(self):
        """Check if FFprobe is available"""
        try:
            result = subprocess.run(['ffprobe', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.debug("✓ FFprobe found")
            else:
                logger.error("⚠️  FFprobe may not be properly installed")
        except FileNotFoundError:
            logger.error("⚠️  FFprobe not found in PATH")
            logger.error("   Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
            raise
        except Exception as e:
            logger.error(f"⚠️  Could not check FFprobe: {e}")
            
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
                        logger.info(f"✓ Calculated duration from {frame_count} frames")
                except:
                    # Final fallback - use a very conservative estimate
                    self.duration = 0
                    
            logger.info(f"✓ Video properties: {self.width}x{self.height}, {self.fps:.2f}fps, {self.duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Error getting video properties: {e}")
            # Fallback to OpenCV for basic properties
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = total_frames / self.fps if self.fps > 0 else 0
                cap.release()
                logger.warning(f"✓ Using OpenCV fallback: {self.width}x{self.height}, {self.fps:.2f}fps, {self.duration:.1f}s")
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
                logger.info(f"✓ Detected 10-bit video format: {pix_fmt}")
                return 10
            else:
                logger.info(f"✓ Detected 8-bit video format: {pix_fmt}")
                return 8
                
        except Exception as e:
            logger.error(f"⚠️  Could not detect bit depth: {e}, assuming 8-bit")
            return 8
        
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
            logger.warning(f"⚠️  Video duration ({self.duration:.1f}s) is shorter than requested analysis period ({start_time}s + {duration}s)")
            if self.duration > start_time:
                duration = self.duration - start_time
                logger.debug(f"   Adjusting duration to {duration:.1f}s")
            else:
                logger.debug(f"   Falling back to first {min(duration, self.duration):.1f}s")
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
            
        # Add signalstats (BRNG only)
        filter_chain += ",signalstats=stat=brng"
        
        # Build FFprobe command
        cmd = [
            'ffprobe',
            '-f', 'lavfi',
            '-i', filter_chain,
            '-show_entries', 'frame_tags=lavfi.signalstats.BRNG',
            '-of', 'csv=p=0'
        ]
        
        logger.debug(f"\nRunning FFprobe signalstats analysis on {region_name}...")
        if active_area:
            x, y, w, h = active_area
            logger.info(f"  Region: {w}x{h} at ({x},{y})")
        else:
            logger.info(f"  Region: full frame {self.width}x{self.height}")
        logger.info(f"  Time range: {start_time}s to {end_time}s ({duration}s duration)")
            
        # Run FFprobe and capture output
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFprobe error: {result.stderr}")
                return None
                
            # Parse CSV output
            return self.parse_ffprobe_output(result.stdout, region_name)
            
        except Exception as e:
            logger.error(f"Error running FFprobe: {e}")
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
                    # CSV output should be just the BRNG value (as a proportion 0-1)
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
            # BRNG values are proportions (0-1), convert to percentages for storage
            results['avg_brng'] = np.mean(results['brng_values']) * 100
            results['max_brng'] = np.max(results['brng_values']) * 100
            
        if results['frames_analyzed'] > 0:
            results['violation_percentage'] = (results['frames_with_violations'] / 
                                             results['frames_analyzed']) * 100
                                             
        logger.info(f"  Analyzed {results['frames_analyzed']} frames")
        logger.warning(f"  Frames with violations: {results['frames_with_violations']} ({results['violation_percentage']:.1f}%)")
        logger.info(f"  Avg BRNG: {results['avg_brng']:.4f}%, Max BRNG: {results['max_brng']:.4f}%")
        
        return results
    
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
            logger.error("No border regions provided")
            return results
        
        for region_name, coords in border_regions.items():
            if coords:
                x, y, w, h = coords
                logger.debug(f"\nAnalyzing {region_name}: {w}x{h} at ({x},{y})")
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
            
            logger.info(f"✓ Loaded border data from {border_data_path}")
            return border_data
        
        except Exception as e:
            logger.error(f"⚠️ Could not load border data: {e}")
            return None


def analyze_video_signalstats(video_path, border_data_path=None, output_dir=None, 
                             start_time=120, duration=60):
    """
    Simplified signalstats analysis focusing on active area and borders only
    
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
        
    logger.debug(f"Processing: {video_path.name}\n")
    logger.info(f"Using FFprobe signalstats BRNG analysis\n")
    logger.debug(f"Analyzing from {start_time//60}:{start_time%60:02d} to {(start_time+duration)//60}:{(start_time+duration)%60:02d}\n")
    
    analyzer = FFprobeAnalyzer(video_path)
    
    # Load border data if provided
    border_data = None
    active_area = None
    
    if border_data_path:
        border_data = analyzer.load_border_data(border_data_path)
        if border_data and border_data.get('active_area'):
            active_area = tuple(border_data['active_area'])
            logger.info(f"Using active area from border data: {active_area}")
    
    # If no border data, analyze full frame as fallback
    if not active_area:
        logger.debug("\nNo border data available - analyzing full frame...")
        full_results = analyzer.analyze_with_ffprobe(
            active_area=None, 
            start_time=start_time,
            duration=duration,
            region_name="full frame"
        )
        active_results = None
        border_results = None
    else:
        # Analyze active area only
        logger.debug("\nAnalyzing active area...")
        active_results = analyzer.analyze_with_ffprobe(
            active_area=active_area,
            start_time=start_time,
            duration=duration,
            region_name="active area"
        )
        
        # Analyze border regions if available
        border_results = None
        if border_data.get('border_regions'):
            logger.debug("\nAnalyzing border regions...")
            border_results = analyzer.analyze_border_regions_from_data(
                border_data['border_regions'],
                start_time=start_time,
                duration=duration
            )
        
        full_results = None  # We're not analyzing full frame when we have border data
    
    # Create report
    report = {
        'video_file': str(video_path),
        'bit_depth': analyzer.bit_depth,
        'analysis_method': 'FFprobe signalstats (simplified)',
        'analysis_period': f'{start_time}s to {start_time + duration}s',
        'sample_duration': duration,
        'border_data_used': border_data_path is not None,
        'active_area': list(active_area) if active_area else None,
        'results': {
            'full_frame': full_results,  # Only present if no border data
            'active_area': active_results,
            'border_regions': border_results
        }
    }
    
    # Analysis and diagnosis
    logger.debug("\n" + "="*80)
    logger.debug("FFPROBE SIGNALSTATS ANALYSIS RESULTS")
    logger.debug("="*80)
    logger.debug(f"\nVideo: {analyzer.bit_depth}-bit")
    logger.info(f"Analysis period: {start_time//60}:{start_time%60:02d} to {(start_time+duration)//60}:{(start_time+duration)%60:02d}")
    
    if active_results:
        logger.info(f"\n🎯 ACTIVE AREA ANALYSIS:")
        logger.debug(f"   Frames with violations: {active_results['frames_with_violations']}/{active_results['frames_analyzed']} ({active_results['violation_percentage']:.1f}%)")
        logger.debug(f"   Average BRNG: {active_results['avg_brng']:.4f}%")
        logger.debug(f"   Maximum BRNG: {active_results['max_brng']:.4f}%")
    elif full_results:
        logger.info(f"\n📺 FULL FRAME ANALYSIS (no border data available):")
        logger.debug(f"   Frames with violations: {full_results['frames_with_violations']}/{full_results['frames_analyzed']} ({full_results['violation_percentage']:.1f}%)")
        logger.debug(f"   Average BRNG: {full_results['avg_brng']:.4f}%")
        logger.debug(f"   Maximum BRNG: {full_results['max_brng']:.4f}%")
    
    if border_results:
        logger.info(f"\n🔴 BORDER REGIONS ANALYSIS:")
        for region, data in border_results.items():
            if data:
                print(f"   {region}: {data['violation_percentage']:.1f}% violations, avg BRNG: {data['avg_brng']:.4f}%")
    
    # Determine diagnosis
    if active_results:
        frame_violation_pct = active_results['violation_percentage']
        avg_brng = active_results['avg_brng']  # Already in percentage
        max_brng = active_results['max_brng']  # Already in percentage
        
        # Check if violations are higher in borders than active area
        border_violation_avg = 0
        border_brng_avg = 0
        if border_results:
            border_violations = [data['violation_percentage'] for data in border_results.values() if data]
            border_brngs = [data['avg_brng'] for data in border_results.values() if data]
            if border_violations:
                border_violation_avg = np.mean(border_violations)
            if border_brngs:
                border_brng_avg = np.mean(border_brngs)
        
        # Diagnosis based on both frame percentage AND severity of violations
        # Consider: percentage of frames with violations AND the actual BRNG values
        
        if frame_violation_pct < 10 and max_brng < 0.01:
            # Very minor violations - less than 10% of frames and less than 0.01% of pixels
            report['diagnosis'] = "✓ Video appears broadcast-compliant"
            logger.info(f"\n✅ DIAGNOSIS: Video appears to be within broadcast range")
            logger.info(f"   Minimal violations: {frame_violation_pct:.1f}% of frames, max {max_brng:.4f}% pixels affected")
            
        elif frame_violation_pct < 50 and max_brng < 0.1:
            # Minor violations - common in professional content
            report['diagnosis'] = "ℹ️ Minor BRNG violations detected - likely acceptable"
            logger.info(f"\n✅ DIAGNOSIS: Minor broadcast range violations detected")
            logger.info(f"   {frame_violation_pct:.1f}% of frames affected, but only {max_brng:.4f}% pixels maximum")
            logger.info("   → These levels are typically acceptable for broadcast")
            
        elif frame_violation_pct > 80 and max_brng < 0.5:
            # Many frames affected but still minor pixel violations
            report['diagnosis'] = "⚠️ Widespread minor BRNG violations"
            logger.info(f"\n⚠️ DIAGNOSIS: Widespread but minor broadcast violations")
            logger.info(f"   {frame_violation_pct:.1f}% of frames affected, up to {max_brng:.4f}% pixels")
            logger.info("   → Review recommended, but may be acceptable depending on content")
            
        elif max_brng > 2.0 or (frame_violation_pct > 50 and max_brng > 1.0):
            # Significant violations - needs correction
            report['diagnosis'] = "⚠️ Significant BRNG violations requiring correction"
            logger.info(f"\n⚠️ DIAGNOSIS: Significant broadcast range violations detected")
            logger.info(f"   {frame_violation_pct:.1f}% of frames with up to {max_brng:.4f}% pixels out of range")
            logger.info("   → Levels adjustment recommended for broadcast compliance")
            
        elif max_brng > 5.0:
            # Severe violations
            report['diagnosis'] = "🔴 Severe BRNG violations"
            logger.info(f"\n🔴 DIAGNOSIS: Severe broadcast range violations")
            logger.info(f"   {frame_violation_pct:.1f}% of frames with up to {max_brng:.4f}% pixels out of range")
            logger.info("   → Video requires levels correction before broadcast")
            
        else:
            # Moderate violations
            report['diagnosis'] = "⚠️ Moderate BRNG violations detected"
            logger.info(f"\n⚠️ DIAGNOSIS: Moderate broadcast range violations")
            logger.info(f"   {frame_violation_pct:.1f}% of frames, max {max_brng:.4f}% pixels affected")
            logger.info("   → Review recommended for broadcast compliance")
            
        # Additional note if borders have significantly more violations
        if border_violation_avg > frame_violation_pct * 2 and border_brng_avg > avg_brng * 2:
            logger.info("   Note: Border regions show higher violation rates than active content")
            
    elif full_results:
        # Fallback diagnosis when no border data available
        frame_violation_pct = full_results['violation_percentage']
        avg_brng = full_results['avg_brng']
        max_brng = full_results['max_brng']
        
        if frame_violation_pct < 10 and max_brng < 0.01:
            report['diagnosis'] = "✓ Video appears broadcast-compliant"
            logger.info(f"\n✅ DIAGNOSIS: Video appears to be within broadcast range")
        elif frame_violation_pct < 50 and max_brng < 0.1:
            report['diagnosis'] = "ℹ️ Minor BRNG violations detected"
            logger.info(f"\n✅ DIAGNOSIS: Minor violations - likely acceptable for broadcast")
        elif max_brng > 2.0 or (frame_violation_pct > 50 and max_brng > 1.0):
            report['diagnosis'] = "⚠️ Significant BRNG violations found"
            logger.info(f"\n⚠️ DIAGNOSIS: Video contains significant broadcast range violations")
        else:
            report['diagnosis'] = "⚠️ Moderate BRNG violations found"
            logger.info(f"\n⚠️ DIAGNOSIS: Video contains moderate broadcast range violations")
    
    # Save JSON report
    json_path = output_dir / f"{video_path.stem}_signalstats_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nDetailed analysis report saved: {json_path}")
    
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