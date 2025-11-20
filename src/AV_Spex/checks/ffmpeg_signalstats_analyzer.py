#!/usr/bin/env python3
"""
FFprobe Signalstats Analyzer (Enhanced with QCTools Integration)

Uses existing QCTools reports when available to pre-screen BRNG violations,
only running FFprobe signalstats when necessary based on QCTools data.
Falls back to direct FFprobe analysis when QCTools reports are unavailable.
"""

import json
import subprocess
import shlex
import numpy as np
from pathlib import Path
import cv2
import gzip
import io
import os
from lxml import etree

from AV_Spex.utils.log_setup import logger


def generate_comprehensive_final_summary(video_id, border_data, signalstats_results, border_results, analysis_decisions):
    """Generate comprehensive summary including all analysis results and smart decisions"""
    
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(f"FRAME ANALYSIS SUMMARY - {video_id}")
    summary_lines.append("=" * 60)
    
    # Border Detection Summary (Robust handling of different structures)
    if border_data:
        method = border_data.get('detection_method', 'sophisticated')
        summary_lines.append(f"Border detection method: {method}")
        
        # Handle active_area - try different possible structures
        if 'active_area' in border_data:
            area = border_data['active_area']
            try:
                if isinstance(area, dict):
                    # Dict structure: {'x': 12, 'y': 5, 'width': 694, 'height': 475}
                    x = area.get('x', 0)
                    y = area.get('y', 0) 
                    w = area.get('width', 0)
                    h = area.get('height', 0)
                elif isinstance(area, (list, tuple)) and len(area) >= 4:
                    # List structure: [12, 5, 694, 475]
                    x, y, w, h = area[0], area[1], area[2], area[3]
                else:
                    # Try to extract from string or other format
                    logger.debug(f"Unexpected active_area format: {area}")
                    x = y = w = h = 0
                
                if w > 0 and h > 0:
                    summary_lines.append(f"Active area: {w}x{h} at ({x},{y})")
            except Exception as e:
                logger.debug("Error parsing active_area: {e}")
        
        # Try to find border widths from multiple possible locations
        border_widths_found = False
        
        # Try border_regions structure
        if 'border_regions' in border_data:
            regions = border_data['border_regions']
            try:
                if 'left_border' in regions:
                    left_data = regions['left_border']
                    if isinstance(left_data, (list, tuple)) and len(left_data) >= 3:
                        summary_lines.append(f"Left border region: {left_data[2]}px wide")
                        border_widths_found = True
                
                if 'right_border' in regions:
                    right_data = regions['right_border']
                    if isinstance(right_data, (list, tuple)) and len(right_data) >= 3:
                        summary_lines.append(f"Right border region: {right_data[2]}px wide")
                        border_widths_found = True
            except Exception as e:
                logger.debug(f"Error parsing border_regions: {e}")
        
        # If not found in border_regions, try other locations
        if not border_widths_found:
            # Look for direct keys
            for key in ['left_border_width', 'right_border_width', 'left_width', 'right_width']:
                if key in border_data:
                    width = border_data[key]
                    border_type = 'Left' if 'left' in key else 'Right'
                    summary_lines.append(f"{border_type} border region: {width}px wide")
                    border_widths_found = True
        
        # Add special findings
        head_switching_keys = ['head_switching_detected', 'head_switching_artifacts', 'head_switching']
        for key in head_switching_keys:
            if border_data.get(key):
                severity = border_data.get('head_switching_severity', 'moderate')
                summary_lines.append(f"⚠️ Head switching artifacts detected - {severity} severity")
                break
    
    # Enhanced Signalstats Summary with Smart Status
    if signalstats_results:
        violation_pct = signalstats_results['violation_percentage']
        max_brng = signalstats_results['max_brng']
        
        # Smart status determination
        if violation_pct < 5 and max_brng < 0.1:
            status_emoji = "✅"
            status_text = "Excellent broadcast compliance"
            action = "No action needed"
        elif violation_pct < 15 and max_brng < 1.0:
            status_emoji = "✅"
            status_text = "Good broadcast compliance"
            action = "Minor violations acceptable"
        elif violation_pct < 35 and max_brng < 2.0:
            status_emoji = "⚠️"
            status_text = "Moderate BRNG violations detected"
            action = "Review recommended for broadcast compliance"
        elif violation_pct < 60 and max_brng < 5.0:
            status_emoji = "⚠️"
            status_text = "Concerning violations"
            action = "Correction likely needed"
        else:
            status_emoji = "❌"
            status_text = "Severe violations"
            action = "Significant correction required"
        
        summary_lines.append(f"Signalstats: {status_emoji} {status_text}")
        summary_lines.append(f"   {violation_pct:.1f}% of frames, max {max_brng:.4f}% pixels affected")
        summary_lines.append(f"   → {action}")
        
        # Show border analysis results if available
        if border_results:
            summary_lines.append("Border analysis:")
            for region_name, data in border_results.items():
                if isinstance(data, dict) and 'violation_percentage' in data and 'avg_brng' in data:
                    summary_lines.append(f"   {region_name}: {data['violation_percentage']:.1f}% violations, "
                                       f"avg {data['avg_brng']:.4f}% BRNG")
    
    # Show analysis optimization decisions
    if analysis_decisions:
        summary_lines.append("")
        summary_lines.append("Analysis optimizations:")
        for decision in analysis_decisions[-5:]:  # Show last 5 decisions
            summary_lines.append(f"   • {decision}")
    
    summary_lines.append("=" * 60)
    
    return "\n".join(summary_lines)


class FFprobeAnalyzer:
    """
    Analyzes video files using FFprobe signalstats for broadcast range compliance
    Enhanced with QCTools report parsing for intelligent analysis decisions
    """
    
    def __init__(self, video_path):
        self.video_path = str(video_path)
        
        # Get video properties
        self.get_video_properties()
        
        # Detect bit depth
        self.bit_depth = self.detect_bit_depth()
        
        # Check for QCTools report
        self.qctools_report_path = self.find_qctools_report()
        
        # Initialize decision tracking
        self._active_area_results = None
        self._border_analysis_decisions = []
        
    def find_qctools_report(self):
        """Find existing QCTools report for this video"""
        video_path = Path(self.video_path)
        video_id = video_path.stem
        
        # Look for QCTools reports in common locations
        search_patterns = [
            # In same directory as video
            video_path.parent / f"{video_id}.qctools.xml.gz",
            video_path.parent / f"{video_id}.qctools.mkv",
            # In metadata subdirectories
            video_path.parent / f"{video_id}_qc_metadata" / f"{video_id}.qctools.xml.gz",
            video_path.parent / f"{video_id}_qc_metadata" / f"{video_id}.mkv.qctools.mkv",
            video_path.parent / f"{video_id}_vrecord_metadata" / f"{video_id}.qctools.xml.gz",
            video_path.parent / f"{video_id}_vrecord_metadata" / f"{video_id}.mkv.qctools.mkv",
        ]
        
        for pattern in search_patterns:
            if pattern.exists():
                # If it's an MKV, we need to extract the XML first
                if str(pattern).endswith('.mkv'):
                    xml_path = self.extract_xml_from_mkv(str(pattern))
                    if xml_path and Path(xml_path).exists():
                        logger.info(f"✓ Found QCTools report (extracted from MKV): {xml_path}\n")
                        return xml_path
                else:
                    logger.info(f"✓ Found QCTools report: {pattern}")
                    return str(pattern)
        
        logger.info("ℹ No existing QCTools report found, will use direct FFprobe analysis")
        return None
    
    def extract_xml_from_mkv(self, mkv_path):
        """Extract qctools.xml.gz from QCTools MKV container"""
        xml_path = mkv_path.replace('.qctools.mkv', '.qctools.xml.gz')
        
        if Path(xml_path).exists():
            return xml_path
            
        try:
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'panic',
                '-dump_attachment:t:0', xml_path,
                '-i', mkv_path
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0 and Path(xml_path).exists():
                return xml_path
        except Exception as e:
            logger.warning(f"Could not extract XML from MKV: {e}")
        
        return None
    
    def parse_qctools_brng(self, start_time, end_time):
        """
        Parse BRNG values from QCTools report for a specific time range
        
        Returns:
            dict: Statistics about BRNG violations in the time range
        """
        if not self.qctools_report_path:
            return None
            
        logger.debug(f"Parsing QCTools BRNG data for {start_time:.1f}s to {end_time:.1f}s")
        
        brng_values = []
        frames_analyzed = 0
        frames_with_violations = 0
        
        try:
            with gzip.open(self.qctools_report_path, 'rb') as gz_file:
                # Create parser
                parser = etree.iterparse(gz_file, events=('end',), tag='frame')
                
                for event, elem in parser:
                    if elem.attrib.get('media_type') == 'video':
                        # Get timestamp - try both pkt_dts_time and pkt_pts_time
                        timestamp_str = elem.attrib.get('pkt_dts_time') or elem.attrib.get('pkt_pts_time')
                        if not timestamp_str:
                            continue
                            
                        timestamp = float(timestamp_str)
                        
                        # Check if within our time range
                        if timestamp >= start_time and timestamp <= end_time:
                            # Find BRNG value in frame tags
                            for tag in elem:
                                if tag.attrib.get('key', '').endswith('BRNG'):
                                    brng_value = float(tag.attrib['value'])
                                    brng_values.append(brng_value)
                                    frames_analyzed += 1
                                    if brng_value > 0:
                                        frames_with_violations += 1
                                    break
                        
                        # Stop if past our time range
                        elif timestamp > end_time:
                            break
                    
                    # Clear element to save memory
                    elem.clear()
                    
        except Exception as e:
            logger.error(f"Error parsing QCTools report: {e}")
            return None
        
        if not brng_values:
            logger.warning("No BRNG data found in QCTools report for specified time range")
            return None
        
        # Calculate statistics (BRNG values in QCTools are 0-1 proportions)
        results = {
            'source': 'qctools',
            'frames_analyzed': frames_analyzed,
            'frames_with_violations': frames_with_violations,
            'brng_values': brng_values,
            'avg_brng': np.mean(brng_values) * 100,  # Convert to percentage
            'max_brng': np.max(brng_values) * 100,
            'min_brng': np.min(brng_values) * 100,
            'violation_percentage': (frames_with_violations / frames_analyzed * 100) if frames_analyzed > 0 else 0
        }
        
        logger.info(f"  QCTools BRNG: {frames_with_violations}/{frames_analyzed} frames with violations ({results['violation_percentage']:.1f}%)")
        logger.info(f"  Max BRNG: {results['max_brng']:.4f}%, Avg: {results['avg_brng']:.4f}%")
        
        return results
    
    def should_run_ffprobe_analysis(self, qctools_results, region_type='full_frame', active_area_results=None):
        """
        Enhanced decision logic for whether to run FFprobe signalstats
        """
        # For border regions, always skip - they're expected to be black
        if region_type in ['left_border', 'right_border', 'top_border', 'bottom_border']:
            return False, "Border regions are expected to violate BRNG (they're black/blanking)"
        
        if not qctools_results:
            return True, "No QCTools data available"
        
        max_brng = qctools_results['max_brng']
        violation_pct = qctools_results['violation_percentage']
        
        # Enhanced thresholds for smarter decisions
        if region_type == 'full_frame':
            # Very strict threshold for full frame - if violations are minimal, skip
            if max_brng < 0.02 and violation_pct < 1:
                return False, f"Negligible violations (QCTools: {violation_pct:.2f}% frames, {max_brng:.4f}% max BRNG)"
            elif max_brng < 0.05 and violation_pct < 3:
                return False, f"Minimal violations (QCTools: {violation_pct:.1f}% frames, {max_brng:.4f}% max BRNG)"
                
        elif region_type == 'active_area':
            # For active area, be slightly more lenient
            if max_brng < 0.02 and violation_pct < 2:
                return False, f"Negligible active area violations (QCTools: {violation_pct:.2f}% frames, {max_brng:.4f}% max BRNG)"
            elif max_brng < 0.1 and violation_pct < 10:
                return False, f"Low active area violations (QCTools: {violation_pct:.1f}% frames, {max_brng:.4f}% max BRNG)"
        
        return True, f"Analysis warranted ({violation_pct:.1f}% violations, {max_brng:.4f}% max BRNG)"
    
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
            logger.error(f"⚠️ Could not detect bit depth: {e}, assuming 8-bit")
            return 8

    def assess_frame_quality_simple(self, frame_time):
        """
        Quick frame quality assessment at a specific time
        Uses simplified version of the border detector's quality assessment
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
            
        frame_idx = int(frame_time * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic quality checks
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Reject very dark/bright frames or low contrast
        if mean_brightness < 15 or mean_brightness > 240 or std_brightness < 15:
            return False
            
        # Check for overly uniform frames
        median_brightness = np.median(gray)
        uniform_pixels = np.sum(np.abs(gray - median_brightness) < 20)
        uniform_percentage = (uniform_pixels / gray.size) * 100
        
        if uniform_percentage > 85:
            return False
            
        return True

    def detect_black_segments(self, min_duration=30, check_interval=3):
        """
        Detect extended black segments in the video with improved detection
        Returns list of (start_time, end_time) tuples for black segments
        """
        logger.debug("Detecting extended black segments...")
        
        black_segments = []
        current_black_start = None
        
        # Check frames at regular intervals - reduced interval for better detection
        total_frames = int(self.duration * self.fps)
        check_frames = range(0, total_frames, int(check_interval * self.fps))
        
        # Ensure we check the full video length
        max_frame_to_check = min(total_frames - 1, int(self.duration * self.fps))
        if max_frame_to_check not in check_frames:
            check_frames = list(check_frames) + [max_frame_to_check]
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.warning("Could not open video for black segment detection")
            return black_segments
        
        logger.debug(f"  Checking {len(check_frames)} frames across {self.duration:.1f}s video")
        
        for i, frame_idx in enumerate(check_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Check if frame is black - more nuanced detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            frame_time = frame_idx / self.fps
            
            # Enhanced black detection criteria
            # Account for different types of black content
            is_very_black = mean_brightness < 3 and std_brightness < 2    # Pure black
            is_near_black = mean_brightness < 8 and std_brightness < 5    # Very dark content
            is_uniform_dark = mean_brightness < 15 and std_brightness < 3  # Uniform dark content
            
            is_black = is_very_black or is_near_black or is_uniform_dark
            
            if is_black:
                if current_black_start is None:
                    current_black_start = frame_time
            else:
                if current_black_start is not None:
                    # End of black segment
                    duration = frame_time - current_black_start
                    if duration >= min_duration:
                        black_segments.append((current_black_start, frame_time))
                        logger.info(f"  Black segment: {current_black_start//60:.0f}:{current_black_start%60:05.2f} - {frame_time//60:.0f}:{frame_time%60:05.2f} ({duration:.1f}s)")
                    current_black_start = None
            
            # Progress logging for long videos
            if i > 0 and i % 100 == 0:
                logger.debug(f"    Scanned to {frame_time//60:.0f}:{frame_time%60:04.1f}")
        
        # Handle case where video ends during black segment
        if current_black_start is not None:
            duration = self.duration - current_black_start
            if duration >= min_duration:
                black_segments.append((current_black_start, self.duration))
                logger.info(f"  Black segment: {current_black_start//60:.0f}:{current_black_start%60:05.2f} - {self.duration//60:.0f}:{self.duration%60:05.2f} ({duration:.1f}s)")
        
        cap.release()
        
        if black_segments:
            logger.info(f"✓ Found {len(black_segments)} extended black segment(s)")
        else:
            logger.debug("  No extended black segments detected")
        
        return black_segments

    def find_good_analysis_periods(self, content_start_time=0, color_bars_end_time=None, 
                                  target_duration=60, num_periods=3, min_gap=30):
        """
        Find multiple good periods throughout the video for analysis, avoiding black segments
        
        Args:
            content_start_time: When actual content starts (from scene detection)
            color_bars_end_time: When color bars end (from qct-parse)
            target_duration: Duration of each analysis period
            num_periods: Number of analysis periods to find
            min_gap: Minimum gap between analysis periods
        """
        # Determine the earliest start time
        effective_start = max(content_start_time, color_bars_end_time or 0)
        
        # Add buffer after color bars/content start
        effective_start += 10  # 10 second buffer
        
        # Detect black segments automatically
        black_segments = self.detect_black_segments()
        
        # Calculate available duration for analysis
        available_duration = self.duration - effective_start - 30  # Leave 30s at end
        
        if available_duration < target_duration:
            logger.warning(f"⚠️ Limited analysis duration available ({available_duration:.1f}s)")
            return [(effective_start, min(target_duration, available_duration))]
        
        logger.debug(f"\nFinding good analysis periods:")
        logger.debug(f"  Content starts: {content_start_time:.1f}s")
        if color_bars_end_time:
            logger.debug(f"  Color bars end: {color_bars_end_time:.1f}s")
        logger.debug(f"  Effective start: {effective_start:.1f}s")
        logger.debug(f"  Available duration: {available_duration:.1f}s")
        
        def overlaps_black_segment(start, end):
            """Check if proposed period overlaps with any black segment"""
            for black_start, black_end in black_segments:
                if not (end <= black_start or start >= black_end):  # Segments overlap
                    return True, (black_start, black_end)
            return False, None
        
        good_periods = []
        
        # Create candidate periods throughout the video
        if num_periods == 1:
            candidates = [effective_start + (available_duration - target_duration) / 2]
        else:
            # Create more candidates than needed so we can filter out black segments
            period_spacing = (available_duration - target_duration) / max(1, (num_periods * 2 - 1))
            period_spacing = max(period_spacing, min_gap)
            
            candidates = []
            current_time = effective_start
            while current_time + target_duration <= self.duration - 30 and len(candidates) < num_periods * 3:
                candidates.append(current_time)
                current_time += period_spacing
        
        # Filter candidates to avoid black segments
        for candidate_start in candidates:
            if len(good_periods) >= num_periods:
                break
                
            candidate_end = candidate_start + target_duration
            
            # Check for black segment overlap
            overlaps, black_seg = overlaps_black_segment(candidate_start, candidate_end)
            
            if overlaps:
                logger.debug(f"  Candidate {candidate_start//60:.0f}:{candidate_start%60:04.1f} overlaps black segment, skipping")
                continue
            
            # Check minimum gap from existing periods
            too_close = False
            for existing_start, existing_duration in good_periods:
                existing_end = existing_start + existing_duration
                if (candidate_start < existing_end + min_gap and 
                    candidate_end > existing_start - min_gap):
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Test frame quality at candidate period
            if self.assess_frame_quality_simple(candidate_start + target_duration/4) and \
               self.assess_frame_quality_simple(candidate_start + 3*target_duration/4):
                good_periods.append((candidate_start, target_duration))
                logger.debug(f"  ✓ Selected period {len(good_periods)}: {candidate_start//60:.0f}:{candidate_start%60:04.1f}")
            else:
                logger.debug(f"  Candidate {candidate_start//60:.0f}:{candidate_start%60:04.1f} failed quality check")
        
        # If we didn't find enough periods, try to find some by being less strict
        if len(good_periods) < num_periods:
            logger.warning(f"⚠️ Only found {len(good_periods)} good periods, looking for additional periods with relaxed criteria")
            
            # Try periods after black segments end
            for black_start, black_end in black_segments:
                if len(good_periods) >= num_periods:
                    break
                    
                # Try starting 10 seconds after black segment ends
                candidate_start = black_end + 10
                candidate_end = candidate_start + target_duration
                
                if candidate_end <= self.duration - 30:
                    # Check if not too close to existing periods
                    too_close = False
                    for existing_start, existing_duration in good_periods:
                        existing_end = existing_start + existing_duration
                        if abs(candidate_start - existing_start) < min_gap:
                            too_close = True
                            break
                    
                    if not too_close:
                        good_periods.append((candidate_start, target_duration))
                        logger.debug(f"  ✓ Added post-black period: {candidate_start//60:.0f}:{candidate_start%60:04.1f}")
        
        if not good_periods:
            # Ultimate fallback - find any non-black period
            logger.warning("⚠️ No good periods found, using fallback strategy")
            fallback_start = max(600, effective_start)  # Try 10 minutes in
            if fallback_start + target_duration <= self.duration - 30:
                overlaps, _ = overlaps_black_segment(fallback_start, fallback_start + target_duration)
                if not overlaps:
                    good_periods = [(fallback_start, target_duration)]
                else:
                    # Last resort - use a period even if it might have some quality issues
                    good_periods = [(effective_start, min(target_duration, available_duration))]
        
        logger.info(f"✓ Selected {len(good_periods)} analysis period(s):")
        for i, (start, duration) in enumerate(good_periods, 1):
            logger.info(f"  Period {i}: {start//60:.0f}:{start%60:04.1f} - {(start+duration)//60:.0f}:{(start+duration)%60:04.1f} ({duration:.1f}s)")
        
        return good_periods

    def find_quality_period_in_range(self, start_time, end_time, duration):
        """
        Find a good quality period within a time range
        """
        if end_time - start_time < duration:
            return None
            
        # Check several candidate start times within the range
        num_checks = min(10, int((end_time - start_time) / 5))  # Check every 5 seconds, up to 10 times
        check_times = np.linspace(start_time, end_time - duration, num_checks)
        
        for check_time in check_times:
            # Test frame quality at the start and middle of the potential period
            if (self.assess_frame_quality_simple(check_time) and 
                self.assess_frame_quality_simple(check_time + duration/2)):
                return check_time
        
        # If no good quality period found, return the middle of the range
        return start_time + (end_time - start_time - duration) / 2
        
    def analyze_with_ffprobe(self, active_area=None, start_time=120, duration=60, region_name="frame"):
        """
        Use FFprobe with lavfi to analyze BRNG statistics
        
        Args:
            active_area: tuple (x, y, w, h) to crop to, or None for full frame
            start_time: start analysis at this time in seconds
            duration: duration of analysis in seconds
            region_name: descriptive name for this region
        """
        # Check if video is long enough
        if self.duration < start_time + duration:
            logger.warning(f"⚠️ Video duration ({self.duration:.1f}s) is shorter than requested analysis period ({start_time}s + {duration}s)")
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
        logger.info(f"  Time range: {start_time:.1f}s to {end_time:.1f}s ({duration:.1f}s duration)")
            
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

    def analyze_multiple_periods(self, active_area=None, analysis_periods=None, region_name="frame"):
        """
        Analyze multiple periods and combine results with enhanced decision tracking
        """
        if not analysis_periods:
            analysis_periods = [(120, 60)]  # Fallback to original behavior
        
        all_results = []
        combined_brng_values = []
        total_frames = 0
        total_violations = 0
        analysis_decisions = []  # Track analysis decisions
        qctools_cache = {}  # Cache QCTools results by time period
        
        for i, (start_time, duration) in enumerate(analysis_periods, 1):
            logger.debug(f"\n--- Analysis Period {i}/{len(analysis_periods)} ---")
            
            # First, try to get data from QCTools if available
            qctools_results = None
            cache_key = f"{start_time:.1f}_{duration:.1f}"
            
            if self.qctools_report_path:
                # Check cache first
                if cache_key in qctools_cache:
                    qctools_results = qctools_cache[cache_key]
                    # Don't log this for border regions to avoid confusion
                    if not region_name.startswith('border'):
                        logger.debug(f"  Using cached QCTools data for period {i}")
                else:
                    end_time = start_time + duration
                    qctools_results = self.parse_qctools_brng(start_time, end_time)
                    if qctools_results:
                        qctools_cache[cache_key] = qctools_results
            
            # Enhanced decision making with active area comparison
            active_area_results = getattr(self, '_active_area_results', None) if region_name.startswith('border') else None
            should_run, reason = self.should_run_ffprobe_analysis(qctools_results, region_name.split('_')[0], active_area_results)
            analysis_decisions.append(f"Period {i}: {reason}")
            
            if should_run:
                # Run FFprobe analysis
                period_result = self.analyze_with_ffprobe(
                    active_area=active_area,
                    start_time=start_time,
                    duration=duration,
                    region_name=f"{region_name}_period_{i}"
                )
            else:
                # Use QCTools results instead
                period_result = qctools_results
                period_result['region_name'] = f"{region_name}_period_{i}"
                logger.info(f"  Using QCTools data instead of FFprobe for period {i}")
            
            if period_result:
                all_results.append(period_result)
                combined_brng_values.extend(period_result['brng_values'])
                total_frames += period_result['frames_analyzed']
                total_violations += period_result['frames_with_violations']
        
        if not all_results:
            return None
            
        # Create combined results with analysis decisions
        combined_result = {
            'region_name': region_name,
            'analysis_periods': len(analysis_periods),
            'frames_analyzed': total_frames,
            'frames_with_violations': total_violations,
            'brng_values': combined_brng_values,
            'period_results': all_results,
            'data_source': 'mixed' if any(r.get('source') == 'qctools' for r in all_results) else 'ffprobe',
            'analysis_decisions': analysis_decisions
        }
        
        # Store active area results for border analysis decisions
        if region_name == "active area":
            self._active_area_results = combined_result
        
        # Calculate combined statistics
        if combined_brng_values:
            combined_result['avg_brng'] = np.mean(combined_brng_values) * 100
            combined_result['max_brng'] = np.max(combined_brng_values) * 100
        else:
            combined_result['avg_brng'] = 0.0
            combined_result['max_brng'] = 0.0
            
        if total_frames > 0:
            combined_result['violation_percentage'] = (total_violations / total_frames) * 100
        else:
            combined_result['violation_percentage'] = 0.0
            
        logger.info(f"\n=== Combined Results for {region_name} ===")
        logger.info(f"  Total frames analyzed: {total_frames} across {len(analysis_periods)} periods")
        logger.warning(f"  Frames with violations: {total_violations} ({combined_result['violation_percentage']:.1f}%)")
        logger.info(f"  Avg BRNG: {combined_result['avg_brng']:.4f}%, Max BRNG: {combined_result['max_brng']:.4f}%")

        # Store the cache for potential reuse
        self._qctools_cache = qctools_cache
        
        return combined_result
            
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
            'violation_percentage': 0.0,
            'source': 'ffprobe'
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
    
    def analyze_border_regions_from_data(self, border_regions, analysis_periods):
        """
        Skip border region analysis - borders are expected to be black and violate BRNG
        """
        logger.info("\nSkipping border region analysis (borders are expected to violate BRNG)")
        return {
            'analysis_decisions': ["Border analysis skipped - borders are expected to be black/blanking"],
            'skipped': True
        }
        
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
        

    def find_good_analysis_periods_with_hints(self, content_start_time=0, color_bars_end_time=None, 
                                            target_duration=60, num_periods=3, min_gap=30,
                                            quality_frame_hints=None):
        """
        Find analysis periods using hints from border detection about good frames.
        
        Args:
            quality_frame_hints: List of (time_seconds, quality_score) tuples from border detection
        """
        # Determine the earliest start time
        effective_start = max(content_start_time, color_bars_end_time or 0)
        effective_start += 10  # 10 second buffer
        
        # If we have quality frame hints from border detection, use them
        if quality_frame_hints and len(quality_frame_hints) >= num_periods:
            logger.debug("Using quality frame hints from border detection")
            
            # Sort hints by quality score
            sorted_hints = sorted(quality_frame_hints, key=lambda x: x[1], reverse=True)
            
            good_periods = []
            used_times = []
            
            for time_hint, quality_score in sorted_hints:
                if len(good_periods) >= num_periods:
                    break
                    
                # Check if this time is far enough from already selected periods
                too_close = False
                for used_time in used_times:
                    if abs(time_hint - used_time) < min_gap + target_duration:
                        too_close = True
                        break
                
                if not too_close and time_hint >= effective_start:
                    # Use this as a center point for an analysis period
                    period_start = max(effective_start, time_hint - target_duration/2)
                    if period_start + target_duration <= self.duration - 30:
                        good_periods.append((period_start, target_duration))
                        used_times.append(time_hint)
                        logger.debug(f"  ✓ Selected period around quality frame at {time_hint:.1f}s (quality: {quality_score:.3f})")
            
            if len(good_periods) >= num_periods:
                return good_periods
        
        # Fall back to original method if hints don't work out
        return self.find_good_analysis_periods(content_start_time, color_bars_end_time, 
                                            target_duration, num_periods, min_gap)


def analyze_video_signalstats(video_path, border_data_path=None, output_dir=None, 
                             content_start_time=0, color_bars_end_time=None,
                             analysis_duration=60, num_analysis_periods=3,
                             quality_frame_hints=None, border_detection_method='unknown'):
    """
    Enhanced signalstats analysis using QCTools reports when available
    
    Args:
        video_path: Path to video file
        border_data_path: Optional path to border data JSON from border_detector.py
        output_dir: Output directory for results
        content_start_time: When actual content starts (from scene detection)
        color_bars_end_time: When color bars end (from qct-parse)
        analysis_duration: Duration of each analysis period in seconds
        num_analysis_periods: Number of analysis periods to use
    
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
    logger.info(f"Using enhanced FFprobe signalstats with QCTools integration\n")
    
    analyzer = FFprobeAnalyzer(video_path)
    
    # Find good analysis periods - use hints if available from sophisticated detection
    if border_detection_method == 'sophisticated' and quality_frame_hints:
        analysis_periods = analyzer.find_good_analysis_periods_with_hints(
            content_start_time=content_start_time,
            color_bars_end_time=color_bars_end_time,
            target_duration=analysis_duration,
            num_periods=num_analysis_periods,
            quality_frame_hints=quality_frame_hints
        )
    else:
        # Original method for simple borders or no hints
        analysis_periods = analyzer.find_good_analysis_periods(
            content_start_time=content_start_time,
            color_bars_end_time=color_bars_end_time,
            target_duration=analysis_duration,
            num_periods=num_analysis_periods
        )
    
    # Load border data if provided
    border_data = None
    active_area = None
    
    if border_data_path:
        border_data = analyzer.load_border_data(border_data_path)
        if border_data and border_data.get('active_area'):
            active_area = tuple(border_data['active_area'])
            logger.info(f"Using active area from border data: {active_area}")
    
    # Analyze based on available data
    if not active_area:
        logger.debug("\nNo border data available - analyzing full frame...")
        full_results = analyzer.analyze_multiple_periods(
            active_area=None, 
            analysis_periods=analysis_periods,
            region_name="full frame"
        )
        active_results = None
        border_results = None
    else:
        # Analyze active area
        logger.debug("\nAnalyzing active area...")
        active_results = analyzer.analyze_multiple_periods(
            active_area=active_area,
            analysis_periods=analysis_periods,
            region_name="active area"
        )
        
        # Analyze border regions if available
        border_results = None
        if border_data.get('border_regions'):
            logger.debug("\nAnalyzing border regions...")
            border_results = analyzer.analyze_border_regions_from_data(
                border_data['border_regions'],
                analysis_periods
            )
        
        full_results = None  # We're not analyzing full frame when we have border data
    
    # Create report
    report = {
        'video_file': str(video_path),
        'bit_depth': analyzer.bit_depth,
        'analysis_method': 'FFprobe signalstats with QCTools integration' if analyzer.qctools_report_path else 'FFprobe signalstats (no QCTools report)',
        'qctools_report_used': analyzer.qctools_report_path is not None,
        'analysis_periods': [{'start': start, 'duration': duration} for start, duration in analysis_periods],
        'scene_detection_used': True,
        'content_start_time': content_start_time,
        'color_bars_end_time': color_bars_end_time,
        'border_data_used': border_data_path is not None,
        'active_area': list(active_area) if active_area else None,
        'results': {
            'full_frame': full_results,  # Only present if no border data
            'active_area': active_results,
            'border_regions': border_results
        }
    }
    
    # Collect all analysis decisions from various sources
    all_analysis_decisions = []
    
    if active_results and 'analysis_decisions' in active_results:
        all_analysis_decisions.extend(active_results['analysis_decisions'])
    
    if border_results:
        for region_data in border_results.values():
            if isinstance(region_data, dict) and 'analysis_decisions' in region_data:
                all_analysis_decisions.extend(region_data['analysis_decisions'])
    
    # Add border analysis decisions if available
    if hasattr(analyzer, '_border_analysis_decisions'):
        all_analysis_decisions.extend(analyzer._border_analysis_decisions)
    
    # Enhanced analysis and diagnosis with comprehensive summary
    logger.debug("\n" + "="*80)
    logger.debug("ENHANCED FFPROBE SIGNALSTATS ANALYSIS RESULTS")
    logger.debug("="*80)
    logger.debug(f"\nVideo: {analyzer.bit_depth}-bit")
    
    if analyzer.qctools_report_path:
        logger.info("📊 Using existing QCTools report for intelligent analysis")
        all_analysis_decisions.append("Using QCTools report for intelligent analysis")

    total_duration = sum(duration for _, duration in analysis_periods)
    logger.info(f"Analysis: {len(analysis_periods)} periods totaling {total_duration:.1f}s")
    
    if color_bars_end_time:
        logger.info(f"Color bars skipped: 0s to {color_bars_end_time:.1f}s")
    if content_start_time > 0:
        logger.info(f"Content starts at: {content_start_time:.1f}s")

    if active_results:
        logger.info(f"\n🎯 ACTIVE AREA ANALYSIS:")
        logger.debug(f"   Frames with violations: {active_results['frames_with_violations']}/{active_results['frames_analyzed']} ({active_results['violation_percentage']:.1f}%)")
        logger.debug(f"   Average BRNG: {active_results['avg_brng']:.4f}%")
        logger.debug(f"   Maximum BRNG: {active_results['max_brng']:.4f}%")
        if active_results.get('data_source') == 'qctools':
            logger.debug("   (Data from QCTools report)")
    elif full_results:
        logger.info(f"\n📺 FULL FRAME ANALYSIS (no border data available):")
        logger.debug(f"   Frames with violations: {full_results['frames_with_violations']}/{full_results['frames_analyzed']} ({full_results['violation_percentage']:.1f}%)")
        logger.debug(f"   Average BRNG: {full_results['avg_brng']:.4f}%")
        logger.debug(f"   Maximum BRNG: {full_results['max_brng']:.4f}%")
        if full_results.get('data_source') == 'qctools':
            logger.debug("   (Data from QCTools report)")

    if border_results:
        logger.info(f"\n🔴 BORDER REGIONS ANALYSIS:")
        # Debug the structure first
        
        for region_name, data in border_results.items():
            if isinstance(data, dict) and 'violation_percentage' in data:
                logger.info(f"   {region_name}: {data['violation_percentage']:.1f}% violations, avg BRNG: {data['avg_brng']:.4f}%")
            elif region_name == 'analysis_decisions':
                continue  # Skip analysis_decisions

    # Generate and display enhanced summary
    video_id = video_path.stem
    enhanced_summary = generate_comprehensive_final_summary(
        video_id, border_data, active_results or full_results, border_results, all_analysis_decisions
    )
    print("\n" + enhanced_summary)
    
    # Determine diagnosis (same logic as before but with enhanced data)
    primary_result = active_results or full_results
    
    if primary_result:
        frame_violation_pct = primary_result['violation_percentage']
        avg_brng = primary_result['avg_brng']
        max_brng = primary_result['max_brng']
        
        if frame_violation_pct < 10 and max_brng < 0.01:
            report['diagnosis'] = "✓ Video appears broadcast-compliant"
            logger.info(f"\n✅ DIAGNOSIS: Video appears to be within broadcast range")
            logger.info(f"   Minimal violations: {frame_violation_pct:.1f}% of frames, max {max_brng:.4f}% pixels affected")
        elif frame_violation_pct < 50 and max_brng < 0.1:
            report['diagnosis'] = "ℹ️ Minor BRNG violations detected - likely acceptable"
            logger.info(f"\n✅ DIAGNOSIS: Minor broadcast range violations detected")
            logger.info(f"   {frame_violation_pct:.1f}% of frames affected, but only {max_brng:.4f}% pixels maximum")
            logger.info("   → These levels are typically acceptable for broadcast")
        elif max_brng > 2.0 or (frame_violation_pct > 50 and max_brng > 1.0):
            report['diagnosis'] = "⚠️ Significant BRNG violations requiring correction"
            logger.info(f"\n⚠️ DIAGNOSIS: Significant broadcast range violations detected")
            logger.info(f"   {frame_violation_pct:.1f}% of frames with up to {max_brng:.4f}% pixels out of range")
            logger.info("   → Levels adjustment recommended for broadcast compliance")
        elif max_brng > 5.0:
            report['diagnosis'] = "🔴 Severe BRNG violations"
            logger.info(f"\n🔴 DIAGNOSIS: Severe broadcast range violations")
            logger.info(f"   {frame_violation_pct:.1f}% of frames with up to {max_brng:.4f}% pixels out of range")
            logger.info("   → Video requires levels correction before broadcast")
        else:
            report['diagnosis'] = "⚠️ Moderate BRNG violations detected"
            logger.info(f"\n⚠️ DIAGNOSIS: Moderate broadcast range violations")
            logger.info(f"   {frame_violation_pct:.1f}% of frames, max {max_brng:.4f}% pixels affected")
            logger.info("   → Review recommended for broadcast compliance")
    
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
    
    # Process with enhanced FFprobe signalstats analysis
    results = analyze_video_signalstats(
        video_file, 
        border_data_path=border_data_file,
        content_start_time=0,  # Would come from scene detection
        color_bars_end_time=37.9,  # Would come from qct-parse
        analysis_duration=60,
        num_analysis_periods=2
    )