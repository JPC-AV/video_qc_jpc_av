#!/usr/bin/env python3
"""
Enhanced Frame Analysis Module
Combines the efficiency of the refactored version with the sophistication of the original implementation.
"""

import os
import json
import gzip
import cv2
import numpy as np
import subprocess
import shlex
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import xml.etree.ElementTree as ET
from scipy import ndimage, signal
import logging

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils.config_setup import ChecksConfig

# Data classes for structured results
@dataclass
class FrameViolation:
    """Represents a frame with BRNG violations"""
    frame_num: int
    timestamp: float
    brng_value: float  # Changed from brng_low and brng_high
    violation_score: float
    violation_pixels: int = 0
    violation_percentage: float = 0.0
    diagnostics: List[str] = None
    pattern_analysis: Dict = None

@dataclass
class BorderDetectionResult:
    """Results from border detection"""
    active_area: Tuple[int, int, int, int]  # x, y, width, height
    border_regions: Dict
    detection_method: str
    quality_frame_hints: List[Tuple[float, float]]  # (timestamp, quality_score)
    head_switching_artifacts: Optional[Dict] = None
    requires_refinement: bool = False
    expansion_recommendations: Dict = None

@dataclass
class BRNGAnalysisResult:
    """Results from BRNG analysis"""
    violations: List[FrameViolation]
    aggregate_patterns: Dict
    actionable_report: Dict
    thumbnails: List[str]
    requires_border_adjustment: bool
    refinement_recommendations: Dict = None

@dataclass
class SignalstatsResult:
    """Results from signalstats analysis"""
    violation_percentage: float
    max_brng: float
    avg_brng: float
    analysis_periods: List[Dict]
    diagnosis: str
    used_qctools: bool


class QCToolsParser:
    """Memory-efficient QCTools parser with streaming capabilities"""
    
    def __init__(self, report_path: str, fps: float = 29.97):
        self.report_path = report_path
        self.fps = fps
        self.bit_depth_10 = self._detect_bit_depth()
        
    def _detect_bit_depth(self) -> bool:
        """Detect if video is 10-bit by checking YMAX values"""
        try:
            if self.report_path.endswith('.gz'):
                file_handle = gzip.open(self.report_path, 'rt')
            else:
                file_handle = open(self.report_path, 'r')
            
            parser = ET.iterparse(file_handle, events=['start', 'end'])
            parser = iter(parser)
            event, root = next(parser)
            
            frame_count = 0
            for event, elem in parser:
                if event == 'end' and elem.tag == 'frame':
                    ymax = float(elem.findtext('.//tag[@key="lavfi.signalstats.YMAX"]', '255'))
                    if ymax > 250:
                        file_handle.close()
                        return True
                    frame_count += 1
                    if frame_count > 100:  # Check first 100 frames
                        break
                    elem.clear()
                    root.clear()
            
            file_handle.close()
            return False
        except:
            return False
    
    def parse_for_violations_streaming_period(self, start_time: float, end_time: float, 
                                        period_num: int, max_frames: int = 100, 
                                        skip_color_bars: bool = True) -> List[FrameViolation]:
        """Stream parse QCTools report for BRNG violations in specific time period"""
        violations = []
        
        # Counters for this specific period
        frames_in_period = 0
        frames_with_violations = 0
        max_brng_value = 0
        frames_checked = 0
        
        try:
            if self.report_path.endswith('.gz'):
                file_handle = gzip.open(self.report_path, 'rt')
            else:
                file_handle = open(self.report_path, 'r')
            
            parser = ET.iterparse(file_handle, events=['start', 'end'])
            parser = iter(parser)
            event, root = next(parser)
            
            for event, elem in parser:
                if event == 'end' and elem.tag == 'frame':
                    frames_checked += 1
                    
                    # Get timestamp from the frame element
                    timestamp_str = elem.get('pkt_pts_time')
                    if not timestamp_str:
                        elem.clear()
                        root.clear()
                        continue
                    
                    timestamp = float(timestamp_str)
                    
                    # Skip frames outside our period
                    if timestamp < start_time:
                        elem.clear()
                        root.clear()
                        continue
                        
                    if timestamp > end_time:
                        elem.clear()
                        root.clear()
                        break  # We've passed our period, stop parsing
                    
                    frames_in_period += 1
                    
                    # Extract frame data
                    frame_data = self._extract_frame_violations(elem, frame_num=None)
                    if frame_data:
                        frames_with_violations += 1
                        max_brng_value = max(max_brng_value, frame_data.brng_value)
                        violations.append(frame_data)
                    
                    elem.clear()
                    root.clear()
                    
                    # Stop if we have enough violations
                    if len(violations) >= max_frames:
                        break
            
            file_handle.close()
            
            # Log period-specific summary
            if frames_in_period > 0:
                violation_pct = (frames_with_violations / frames_in_period * 100)
                logger.info(f"    Period {period_num}: {frames_in_period:,} frames analyzed, "
                        f"{frames_with_violations:,} with violations ({violation_pct:.1f}%)")
                if frames_with_violations > 0:
                    logger.info(f"    Period {period_num} max BRNG: {max_brng_value:.4f}%")
            else:
                logger.info(f"    Period {period_num}: No frames found in time range {start_time:.1f}s - {end_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Error parsing QCTools report for period {period_num}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        violations.sort(key=lambda x: x.violation_score, reverse=True)
        return violations[:max_frames]
    
    def parse_for_violations_streaming(self, max_frames: int = 100, 
                                 skip_color_bars: bool = True,
                                 color_bars_end_time: float = 0) -> List[FrameViolation]:
        """Stream parse QCTools report for BRNG violations"""
        violations = []
        chunk_size = 1000
        
        # Counters
        total_frames_checked = 0
        frames_after_color_bars = 0
        frames_with_violations = 0
        frames_skipped = 0
        max_brng_value = 0
        
        try:
            if self.report_path.endswith('.gz'):
                file_handle = gzip.open(self.report_path, 'rt')
            else:
                file_handle = open(self.report_path, 'r')
            
            parser = ET.iterparse(file_handle, events=['start', 'end'])
            parser = iter(parser)
            event, root = next(parser)
            
            frame_buffer = []
            
            for event, elem in parser:
                if event == 'end' and elem.tag == 'frame':
                    total_frames_checked += 1
                    
                    # Get timestamp from the frame element
                    timestamp_str = elem.get('pkt_pts_time')
                    if not timestamp_str:
                        elem.clear()
                        root.clear()
                        continue
                        
                    timestamp = float(timestamp_str)
                    
                    # Skip color bars based on timestamp
                    if skip_color_bars and color_bars_end_time > 0 and timestamp < color_bars_end_time:
                        frames_skipped += 1
                        elem.clear()
                        root.clear()
                        continue
                    
                    frames_after_color_bars += 1
                    
                    # Extract frame data - pass None for frame_num to let it extract from element
                    frame_data = self._extract_frame_violations(elem, frame_num=None)
                    if frame_data:
                        frames_with_violations += 1
                        max_brng_value = max(max_brng_value, frame_data.brng_value)
                        frame_buffer.append(frame_data)
                    
                    elem.clear()
                    root.clear()
                    
                    # Process buffer when full
                    if len(frame_buffer) >= chunk_size:
                        violations.extend(self._process_violation_buffer(frame_buffer))
                        frame_buffer = []
                        
                        if len(violations) > max_frames * 2:
                            violations.sort(key=lambda x: x.violation_score, reverse=True)
                            violations = violations[:max_frames]
            
            # Process remaining buffer
            if frame_buffer:
                violations.extend(self._process_violation_buffer(frame_buffer))
            
            file_handle.close()
            
            # Log summary
            logger.info(f"  Checked {total_frames_checked:,} frames from QCTools report")
            if frames_skipped > 0:
                logger.info(f"  Skipped {frames_skipped:,} color bar frames (first {color_bars_end_time:.1f}s)")
            
            if frames_after_color_bars > 0:
                violation_pct = (frames_with_violations / frames_after_color_bars * 100) if frames_after_color_bars > 0 else 0
                logger.info(f"  Analyzed {frames_after_color_bars:,} content frames")
                logger.info(f"  Found {frames_with_violations:,} frames with BRNG violations ({violation_pct:.1f}% of content)")
                if frames_with_violations > 0:
                    logger.info(f"  Max BRNG value: {max_brng_value:.4f}%")
            else:
                logger.warning("  No frames analyzed after color bars - check color bars duration")
            
        except Exception as e:
            logger.error(f"Error parsing QCTools report: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        violations.sort(key=lambda x: x.violation_score, reverse=True)
        return violations[:max_frames]
    
    def _extract_frame_violations(self, elem, frame_num: int = None) -> Optional[FrameViolation]:
        """Extract violation data from frame element"""
        try:
            # Get frame number from element if not provided
            if frame_num is None:
                # Try different attributes for frame number
                frame_num_str = elem.get('n') or elem.get('pkt_pts')
                if frame_num_str:
                    frame_num = int(frame_num_str)
                else:
                    return None
            
            # Get timestamp - try pkt_pts_time first, then calculate from frame number
            timestamp_str = elem.get('pkt_pts_time')
            if timestamp_str:
                timestamp = float(timestamp_str)
            else:
                timestamp = frame_num / self.fps if frame_num else 0
            
            # Find BRNG value
            brng_value = None
            brng_tag = elem.find('.//tag[@key="lavfi.signalstats.BRNG"]')
            if brng_tag is not None:
                brng_str = brng_tag.get('value')
                if brng_str:
                    brng_value = float(brng_str)
            
            # Check if this is a violation (BRNG > 0.01 means > 1% of pixels out of range)
            if brng_value is not None and brng_value > 0.01:
                return FrameViolation(
                    frame_num=frame_num,
                    timestamp=timestamp,
                    brng_value=brng_value * 100,  # Convert to percentage
                    violation_score=brng_value
                )
                
        except Exception as e:
            if not hasattr(self, '_logged_extraction_error'):
                logger.debug(f"Error extracting frame violations: {e}")
                self._logged_extraction_error = True
        
        return None
    
    def _process_violation_buffer(self, buffer: List[FrameViolation]) -> List[FrameViolation]:
        """Process buffer of violations"""
        return [v for v in buffer if v is not None]


class SophisticatedBorderDetector:
    """Advanced border detection with quality assessment and refinement capabilities"""
    
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self._init_video_properties()
        
    def _init_video_properties(self):
        """Initialize video properties"""
        cap = cv2.VideoCapture(self.video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()
    
    def detect_borders_with_quality_assessment(self, 
                                              violations: List[FrameViolation] = None,
                                              method: str = 'sophisticated') -> BorderDetectionResult:
        """
        Detect borders using sophisticated quality assessment or simple method.
        
        Args:
            violations: List of frames with known violations for focused detection
            method: 'sophisticated' or 'simple'
        """
        if method == 'simple':
            return self._detect_simple_borders()
        else:
            return self._detect_sophisticated_borders(violations)
    
    def _detect_simple_borders(self, border_size: int = 25) -> BorderDetectionResult:
        """Simple fixed-size border detection"""
        active_x = border_size
        active_y = border_size
        active_width = self.width - (2 * border_size)
        active_height = self.height - (2 * border_size)
        
        # Ensure valid dimensions
        if active_width <= 0 or active_height <= 0:
            active_x = active_y = 0
            active_width = self.width
            active_height = self.height
            border_size = 0
        
        border_regions = self._calculate_border_regions(
            active_x, active_y, active_width, active_height
        )
        
        return BorderDetectionResult(
            active_area=(active_x, active_y, active_width, active_height),
            border_regions=border_regions,
            detection_method='simple',
            quality_frame_hints=[],
            requires_refinement=False
        )
    
    def _detect_sophisticated_borders(self, 
                                     violations: List[FrameViolation] = None) -> BorderDetectionResult:
        """Sophisticated border detection with quality assessment"""
        cap = cv2.VideoCapture(self.video_path)
        
        # Select quality frames for analysis
        quality_frames = self._select_quality_frames(cap, violations)
        
        if len(quality_frames) < 5:
            logger.warning("Insufficient quality frames, falling back to simple detection")
            cap.release()
            return self._detect_simple_borders()
        
        # Detect borders using quality frames
        borders = self._analyze_borders_from_frames(cap, quality_frames)
        
        # Detect head switching artifacts
        head_switching = self._detect_head_switching(cap, borders)
        
        # Check for vertical blanking lines
        vertical_blanking = self._detect_vertical_blanking(cap, quality_frames)
        
        # Adjust borders based on blanking lines
        if vertical_blanking:
            borders = self._adjust_for_blanking(borders, vertical_blanking)
        
        cap.release()
        
        # Calculate active area
        active_x = borders['left']
        active_y = borders['top']
        active_width = self.width - borders['left'] - borders['right']
        active_height = self.height - borders['top'] - borders['bottom']

        # Log the detection results
        logger.info(f"  Detected borders - L:{borders['left']}px R:{borders['right']}px T:{borders['top']}px B:{borders['bottom']}px")
        logger.info(f"  Active picture area: {active_width}x{active_height} (from {self.width}x{self.height})")
        logger.info(f"  Using {len(quality_frames)} quality frames for detection")
        
        # Add padding for safety
        padding = 5
        active_x += padding
        active_y += padding
        active_width -= 2 * padding
        active_height -= 2 * padding
        
        border_regions = self._calculate_border_regions(
            active_x, active_y, active_width, active_height
        )
        
        # Generate quality hints for signalstats
        quality_hints = [(f['timestamp'], f['quality']) for f in quality_frames[:10]]
        
        return BorderDetectionResult(
            active_area=(active_x, active_y, active_width, active_height),
            border_regions=border_regions,
            detection_method='sophisticated',
            quality_frame_hints=quality_hints,
            head_switching_artifacts=head_switching,
            requires_refinement=False
        )
    
    def _select_quality_frames(self, cap, violations: List[FrameViolation] = None) -> List[Dict]:
        """Select high-quality frames for border detection"""
        quality_frames = []
        
        # If we have violations, prioritize those frames
        if violations:
            for v in violations[:30]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_num)
                ret, frame = cap.read()
                if ret:
                    quality = self._assess_frame_quality(frame)
                    if quality['is_suitable']:
                        quality_frames.append({
                            'frame_num': v.frame_num,
                            'timestamp': v.timestamp,
                            'frame': frame,
                            'quality': quality['overall_quality']
                        })
        
        # If we need more frames, sample evenly
        if len(quality_frames) < 30:
            sample_indices = np.linspace(0, self.total_frames - 1, 50, dtype=int)
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    quality = self._assess_frame_quality(frame)
                    if quality['is_suitable']:
                        quality_frames.append({
                            'frame_num': idx,
                            'timestamp': idx / self.fps,
                            'frame': frame,
                            'quality': quality['overall_quality']
                        })
        
        # Sort by quality
        quality_frames.sort(key=lambda x: x['quality'], reverse=True)
        return quality_frames[:30]
    
    def _assess_frame_quality(self, frame) -> Dict:
        """Assess frame quality for suitability"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        result = {
            'is_suitable': False,
            'overall_quality': 0.0,
            'reasons_rejected': []
        }
        
        # Check brightness
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        if mean_brightness < 15:
            result['reasons_rejected'].append("Too dark")
        elif mean_brightness > 240:
            result['reasons_rejected'].append("Too bright")
        elif std_brightness < 15:
            result['reasons_rejected'].append("Low contrast")
        else:
            # Calculate quality score
            brightness_score = 1.0 - abs(mean_brightness - 120) / 120
            contrast_score = min(std_brightness / 50.0, 1.0)
            result['overall_quality'] = (brightness_score + contrast_score) / 2
            result['is_suitable'] = True
        
        return result
    
    def _analyze_borders_from_frames(self, cap, quality_frames: List[Dict]) -> Dict:
        """Analyze borders from quality frames"""
        borders = {'left': [], 'right': [], 'top': [], 'bottom': []}
        threshold = 10
        edge_sample_width = 100
        
        for frame_data in quality_frames:
            frame = frame_data['frame']
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Detect left border
            for x in range(min(edge_sample_width, w)):
                if np.mean(gray[:, x]) > threshold:
                    borders['left'].append(x)
                    break
            
            # Detect right border
            for x in range(w-1, max(w-edge_sample_width-1, -1), -1):
                if np.mean(gray[:, x]) > threshold:
                    borders['right'].append(w - x - 1)
                    break
            
            # Detect top border
            for y in range(min(20, h)):
                if np.mean(gray[y, w//3:2*w//3]) > threshold:
                    borders['top'].append(y)
                    break
            
            # Detect bottom border
            for y in range(h-1, max(h-20, -1), -1):
                if np.mean(gray[y, w//3:2*w//3]) > threshold:
                    borders['bottom'].append(h - y - 1)
                    break
        
        # Calculate median borders
        return {
            'left': int(np.median(borders['left'])) if borders['left'] else 0,
            'right': int(np.median(borders['right'])) if borders['right'] else 0,
            'top': int(np.median(borders['top'])) if borders['top'] else 0,
            'bottom': int(np.median(borders['bottom'])) if borders['bottom'] else 0
        }
    
    def _detect_vertical_blanking(self, cap, quality_frames: List[Dict]) -> Optional[Dict]:
        """Detect vertical blanking lines"""
        blanking = {'left': None, 'right': None}
        edge_width = 30
        
        for frame_data in quality_frames[:10]:
            frame = frame_data['frame']
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Check left edge
            left_region = gray[:, :edge_width]
            for x in range(edge_width):
                column = left_region[:, x]
                if np.mean(column) < 20 and np.std(column) < 10:
                    if blanking['left'] is None or x > blanking['left']:
                        blanking['left'] = x
            
            # Check right edge
            right_region = gray[:, -edge_width:]
            for x in range(edge_width):
                column = right_region[:, x]
                if np.mean(column) < 20 and np.std(column) < 10:
                    if blanking['right'] is None or (w - edge_width + x) < blanking['right']:
                        blanking['right'] = w - edge_width + x
        
        return blanking if any(blanking.values()) else None
    
    def _adjust_for_blanking(self, borders: Dict, blanking: Dict) -> Dict:
        """Adjust borders based on detected blanking lines"""
        if blanking.get('left') and blanking['left'] > borders['left']:
            borders['left'] = blanking['left'] + 2
        if blanking.get('right') and blanking['right'] < self.width - borders['right']:
            borders['right'] = self.width - blanking['right'] + 2
        return borders
    
    def _detect_head_switching(self, cap, borders: Dict) -> Optional[Dict]:
        """Detect head switching artifacts at bottom of frame"""
        # Sample frames for analysis
        sample_frames = np.linspace(0, self.total_frames - 1, 20, dtype=int)
        artifact_count = 0
        
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Analyze bottom 15 lines
            bottom_region = gray[-15:, :]
            
            # Check for asymmetry (characteristic of head switching)
            for line in bottom_region:
                left_half = line[:len(line)//2]
                right_half = line[len(line)//2:]
                
                left_mean = np.mean(left_half)
                right_mean = np.mean(right_half)
                
                if left_mean > 10:
                    asymmetry = abs(left_mean - right_mean) / left_mean
                    if asymmetry > 0.5:
                        artifact_count += 1
                        break
        
        if artifact_count > len(sample_frames) * 0.2:
            return {
                'detected': True,
                'severity': 'high' if artifact_count > len(sample_frames) * 0.5 else 'moderate',
                'percentage': (artifact_count / len(sample_frames)) * 100
            }
        
        return None
    
    def _calculate_border_regions(self, x: int, y: int, w: int, h: int) -> Dict:
        """Calculate border region coordinates"""
        regions = {}
        
        if x > 10:
            regions['left_border'] = (0, 0, x, self.height)
        else:
            regions['left_border'] = None
        
        right_start = x + w
        if right_start < self.width - 10:
            regions['right_border'] = (right_start, 0, self.width - right_start, self.height)
        else:
            regions['right_border'] = None
        
        if y > 10:
            regions['top_border'] = (0, 0, self.width, y)
        else:
            regions['top_border'] = None
        
        bottom_start = y + h
        if bottom_start < self.height - 10:
            regions['bottom_border'] = (0, bottom_start, self.width, self.height - bottom_start)
        else:
            regions['bottom_border'] = None
        
        return regions
    
    def refine_borders(self, current_borders: BorderDetectionResult, 
                       brng_results: BRNGAnalysisResult) -> BorderDetectionResult:
        """
        Refine borders based on BRNG analysis results.
        This implements the smart expansion logic from the original.
        """
        if not brng_results.requires_border_adjustment:
            return current_borders
        
        x, y, w, h = current_borders.active_area
        expansions = brng_results.refinement_recommendations or {}
        
        # Apply recommended expansions
        new_x = max(0, x + expansions.get('left', 0))
        new_y = max(0, y + expansions.get('top', 0))
        new_w = max(100, w - expansions.get('left', 0) - expansions.get('right', 0))
        new_h = max(100, h - expansions.get('top', 0) - expansions.get('bottom', 0))
        
        # Ensure within video bounds
        new_w = min(new_w, self.width - new_x)
        new_h = min(new_h, self.height - new_y)
        
        # Update border regions
        border_regions = self._calculate_border_regions(new_x, new_y, new_w, new_h)
        
        return BorderDetectionResult(
            active_area=(new_x, new_y, new_w, new_h),
            border_regions=border_regions,
            detection_method=current_borders.detection_method + '_refined',
            quality_frame_hints=current_borders.quality_frame_hints,
            head_switching_artifacts=current_borders.head_switching_artifacts,
            requires_refinement=False,
            expansion_recommendations=expansions
        )


class DifferentialBRNGAnalyzer:
    """
    BRNG analyzer using differential detection method.
    Compares highlighted vs original frames to eliminate false positives.
    """
    
    def __init__(self, video_path: str, border_data: BorderDetectionResult = None):
        self.video_path = Path(video_path)
        self.border_data = border_data
        self.active_area = border_data.active_area if border_data else None
        self._init_video_properties()
        
    def _init_video_properties(self):
        """Initialize video properties"""
        cap = cv2.VideoCapture(str(self.video_path))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()
    

    def analyze_with_differential_detection(self, 
                                       output_dir: Path,
                                       duration_limit: int = 300,
                                       skip_start_seconds: float = 0,
                                       qctools_violations: List[FrameViolation] = None,
                                       analysis_periods: List[Tuple[float, int]] = None) -> BRNGAnalysisResult:
        """
        Perform differential BRNG detection by creating highlighted and original versions.
        Now supports analyzing specific periods from signalstats.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Store paths to temporary videos for thumbnail creation
        temp_video_paths = []

        # Use analysis periods if provided, otherwise fall back to original behavior
        if analysis_periods:
            logger.info(f"  Using {len(analysis_periods)} analysis periods from signalstats")
            all_violations = []
            
            for i, (start_time, duration) in enumerate(analysis_periods):
                logger.info(f"  Analyzing period {i+1}: {start_time:.1f}s - {start_time+duration:.1f}s")
                
                # Create temporary directory for this period
                temp_dir = output_dir / f"temp_brng_period_{i+1}"
                temp_dir.mkdir(exist_ok=True)
                
                # Generate comparison videos for this period
                highlighted_path = temp_dir / f"{self.video_path.stem}_highlighted_p{i+1}.mp4"
                original_path = temp_dir / f"{self.video_path.stem}_original_p{i+1}.mp4"
                
                if not self._create_comparison_videos_for_period(
                    highlighted_path, original_path, start_time, duration):
                    logger.error(f"Failed to create comparison videos for period {i+1}")
                    continue
                
                # Store paths for later thumbnail creation
                temp_video_paths.append({
                    'highlighted': highlighted_path,
                    'original': original_path,
                    'start_time': start_time,
                    'duration': duration,
                    'temp_dir': temp_dir
                })
                
                # Analyze violations for this period
                period_violations = self._analyze_differential_violations(
                    highlighted_path, original_path, start_time,
                    qctools_violations=qctools_violations
                )
                
                all_violations.extend(period_violations)
            
            violations = all_violations
            logger.info(f"  Analyzed {len(violations)} frames with potential violations across all periods")
        else:
            # Original single-period analysis (similar handling)
            logger.info(f"  Creating temporary comparison videos (duration: {duration_limit}s)")
            
            temp_dir = output_dir / "temp_brng"
            temp_dir.mkdir(exist_ok=True)
            
            highlighted_path = temp_dir / f"{self.video_path.stem}_highlighted.mp4"
            original_path = temp_dir / f"{self.video_path.stem}_original.mp4"
            
            if not self._create_comparison_videos(highlighted_path, original_path, 
                                                duration_limit, skip_start_seconds):
                logger.error("Failed to create comparison videos")
                return None
            
            temp_video_paths.append({
                'highlighted': highlighted_path,
                'original': original_path,
                'start_time': skip_start_seconds,
                'duration': duration_limit,
                'temp_dir': temp_dir
            })
            
            violations = self._analyze_differential_violations(
                highlighted_path, original_path, skip_start_seconds,
                qctools_violations=qctools_violations
            )
        
        # Generate patterns and reports
        aggregate_patterns = self._analyze_aggregate_patterns(violations)
        actionable_report = self._generate_actionable_report(violations, aggregate_patterns)
        
        # Create thumbnails using temporal diversity selection
        thumbnails = []
        if violations and len(violations) > 0:
            thumb_dir = output_dir / "brng_thumbnails"
            thumb_dir.mkdir(exist_ok=True)
            
            # Select violations with temporal spacing
            selected_violations = self._select_diverse_violations_for_thumbnails(
                violations, 
                max_thumbnails=5, 
                min_time_separation=5.0
            )
            
            logger.info(f"  Creating diagnostic thumbnails for {len(selected_violations)} temporally diverse violations")
            thumbnails = self._create_diagnostic_thumbnails(selected_violations, temp_video_paths, thumb_dir)
            logger.info(f"  Saved {len(thumbnails)} thumbnails to {thumb_dir}")
        
        # Clean up all temporary files
        for video_info in temp_video_paths:
            try:
                video_info['highlighted'].unlink()
                video_info['original'].unlink()
                video_info['temp_dir'].rmdir()
            except:
                pass
        
        return BRNGAnalysisResult(
            violations=violations,
            aggregate_patterns=aggregate_patterns,
            actionable_report=actionable_report,
            thumbnails=thumbnails,
            requires_border_adjustment=aggregate_patterns.get('requires_border_adjustment', False),
            refinement_recommendations=aggregate_patterns.get('expansion_recommendations')
        )
    
    def _create_comparison_videos_for_period(self, highlighted_path: Path, original_path: Path,
                                            start_time: float, duration: int) -> bool:
        """Create highlighted and original versions for a specific time period"""
        # Build crop filter if active area exists
        crop_filter = ""
        if self.active_area:
            x, y, w, h = self.active_area
            crop_filter = f"crop={w}:{h}:{x}:{y},"
        
        # Create highlighted version for this period
        highlighted_cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", str(self.video_path),
            "-t", str(duration),
            "-vf", f"{crop_filter}signalstats=out=brng:color=magenta",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            str(highlighted_path)
        ]
        
        # Create original version for this period
        original_cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", str(self.video_path),
            "-t", str(duration),
            "-vf", crop_filter.rstrip(',') if crop_filter else "null",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            str(original_path)
        ]
        
        try:
            subprocess.run(highlighted_cmd, capture_output=True, check=True)
            subprocess.run(original_cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e}")
            return False
    
    def _analyze_differential_violations(self, highlighted_path: Path, 
                        original_path: Path,
                        period_start_time: float,  # Renamed for clarity
                        qctools_violations: List[FrameViolation] = None) -> List[FrameViolation]:
        """Analyze violations using differential detection"""
        violations = []
        
        cap_h = cv2.VideoCapture(str(highlighted_path))
        cap_o = cv2.VideoCapture(str(original_path))
        
        # Get video properties
        total_frames = int(cap_h.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_h.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate the time range this video segment represents
        period_end_time = period_start_time + duration
        
        # Use QCTools violations to target specific frames
        if qctools_violations and len(qctools_violations) > 0:
            logger.info(f"  Targeting {len(qctools_violations)} frames identified by QCTools")
            
            sample_indices = []
            for v in qctools_violations[:500]:  # Increased limit
                # Check if this violation is within our period
                if period_start_time <= v.timestamp < period_end_time:
                    # Convert to frame position in the extracted video segment
                    relative_time = v.timestamp - period_start_time
                    frame_in_segment = int(relative_time * fps)
                    if 0 <= frame_in_segment < total_frames:
                        sample_indices.append(frame_in_segment)
            
            logger.info(f"  Mapped {len(sample_indices)} violation frames to processed video positions")
            
            # If we didn't get enough samples from QCTools, add some distributed samples
            if len(sample_indices) < 50:
                logger.info(f"  Adding distributed samples to reach minimum coverage")
                additional_samples = np.linspace(0, total_frames - 1, 100, dtype=int)
                for sample in additional_samples:
                    if sample not in sample_indices:
                        sample_indices.append(sample)
                sample_indices = sorted(sample_indices)[:200]
        else:
            # Fallback to distributed sampling
            logger.info(f"  No QCTools violations provided, using distributed sampling")
            num_samples = min(500, total_frames)
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
        
        logger.info(f"  Analyzing {len(sample_indices)} frame samples...")
        
        frames_checked = 0
        for idx in sample_indices:
            cap_h.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_o.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            ret_h, frame_h = cap_h.read()
            ret_o, frame_o = cap_o.read()
            
            if not ret_h or not ret_o:
                continue
            
            frames_checked += 1
            
            # Detect violations differentially with lower threshold
            violation_mask = self._detect_differential_violations_frame(frame_h, frame_o)
            violation_pixels = int(np.sum(violation_mask > 0))
            
            # Lower threshold for detection (was 5, now 2)
            if violation_pixels > 2:
                # Analyze patterns
                pattern_analysis = self._analyze_violation_patterns(violation_mask, frame_o)
                
                # Calculate actual timestamp in original video
                actual_timestamp = (idx / fps) + period_start_time
                
                # Calculate BRNG value from violation pixels
                total_pixels = frame_h.shape[0] * frame_h.shape[1]
                brng_value = (violation_pixels / total_pixels) * 100
                
                violation = FrameViolation(
                    frame_num=int(actual_timestamp * fps),  # Frame number in original video
                    timestamp=actual_timestamp,
                    brng_value=brng_value,
                    violation_score=violation_pixels / total_pixels,
                    violation_pixels=violation_pixels,
                    violation_percentage=brng_value,
                    diagnostics=pattern_analysis.get('diagnostics', []),
                    pattern_analysis=pattern_analysis
                )
                violations.append(violation)
        
        cap_h.release()
        cap_o.release()
        
        # Sort by violation score
        violations.sort(key=lambda x: x.violation_score, reverse=True)
        
        logger.info(f"  Checked {frames_checked} frames, found {len(violations)} with violations above threshold")
        
        return violations
    
    def _detect_differential_violations_frame(self, highlighted: np.ndarray, 
                                     original: np.ndarray, 
                                     sensitivity: str = 'normal') -> np.ndarray:
        """
        Detect violations by comparing highlighted vs original frame
        
        Args:
            sensitivity: 'strict', 'normal', or 'visualization' 
                    - 'strict': Conservative detection for analysis
                    - 'normal': Balanced detection 
                    - 'visualization': More sensitive for thumbnail display
        """
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'strict':
            magenta_threshold = 12
            min_change = 10
            voting_threshold = 2  # Require 2/3 methods
            min_component_size = 15
            morph_iterations = 2
        elif sensitivity == 'visualization':
            magenta_threshold = 6
            min_change = 5
            voting_threshold = 1  # Require only 1/3 methods
            min_component_size = 3
            morph_iterations = 1
        else:  # normal
            magenta_threshold = 10
            min_change = 8
            voting_threshold = 2
            min_component_size = 10
            morph_iterations = 2
        
        # Split channels for analysis
        h_b, h_g, h_r = cv2.split(highlighted)
        o_b, o_g, o_r = cv2.split(original)
        
        # Calculate channel differences
        blue_diff = h_b.astype(np.float32) - o_b.astype(np.float32)
        red_diff = h_r.astype(np.float32) - o_r.astype(np.float32)
        green_diff = h_g.astype(np.float32) - o_g.astype(np.float32)
        
        # Method 1: Strict magenta detection
        strict_magenta = (
            (blue_diff > magenta_threshold) & 
            (red_diff > magenta_threshold) & 
            (green_diff < magenta_threshold/2)
        )
        
        # Method 2: Ratio-based detection
        significant_change = (np.abs(blue_diff) > min_change) | (np.abs(red_diff) > min_change)
        safe_blue = np.where(np.abs(blue_diff) > 0.1, blue_diff, 1)
        safe_red = np.where(np.abs(red_diff) > 0.1, red_diff, 1)
        
        rb_ratio = np.minimum(blue_diff/safe_red, red_diff/safe_blue)
        ratio_tolerance = 0.4
        
        ratio_based = significant_change & (rb_ratio > (1 - ratio_tolerance)) & (rb_ratio < (1 + ratio_tolerance))
        ratio_based = ratio_based & (blue_diff > 0) & (red_diff > 0)
        
        # Method 3: HSV-based magenta detection
        highlighted_hsv = cv2.cvtColor(highlighted, cv2.COLOR_BGR2HSV)
        original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        
        h_h, h_s, h_v = cv2.split(highlighted_hsv)
        o_h, o_s, o_v = cv2.split(original_hsv)
        
        magenta_hue_low = 140
        magenta_hue_high = 160
        near_zero_threshold = 10
        near_180_threshold = 170
        
        is_magenta_hue = (
            ((h_h >= magenta_hue_low) & (h_h <= magenta_hue_high)) |
            (h_h < near_zero_threshold) |
            (h_h > near_180_threshold)
        )
        
        sat_increase = (h_s.astype(np.float32) - o_s.astype(np.float32)) > 15
        value_stable = (h_v.astype(np.float32) - o_v.astype(np.float32)) > -10
        hsv_magenta = is_magenta_hue & sat_increase & value_stable
        
        # Combine methods with voting
        method1 = strict_magenta.astype(np.uint8)
        method2 = ratio_based.astype(np.uint8)
        method3 = hsv_magenta.astype(np.uint8)
        
        vote_sum = method1 + method2 + method3
        combined_mask = (vote_sum >= voting_threshold).astype(np.uint8) * 255
        
        # Morphological operations (less aggressive for visualization)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        
        if sensitivity != 'strict':
            # Skip closing for more sensitive detection
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Component filtering
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
        final_mask = np.zeros_like(cleaned_mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_component_size:
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
                
                # More lenient aspect ratio check for visualization
                min_aspect = 0.05 if sensitivity == 'visualization' else 0.1
                min_area_override = 25 if sensitivity == 'visualization' else 50
                
                if aspect_ratio > min_aspect or area > min_area_override:
                    final_mask[labels == i] = 255
        
        return final_mask

    def _analyze_violation_patterns(self, violation_mask: np.ndarray,
                               frame: np.ndarray) -> Dict:
        """
        Enhanced pattern analysis adapted from ActiveAreaBrngAnalyzer.
        Provides edge blanking, linear patterns, and luma zone diagnostics.
        """
        h, w = violation_mask.shape

        # Edge analysis with linear pattern detection
        edge_violations = self._detect_edge_violations_enhanced(violation_mask, edge_width=15)

        # Edge-based spatial analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edge_violations_mask = cv2.bitwise_and(violation_mask, edges)
        edge_violation_ratio = np.sum(edge_violations_mask > 0) / max(1, np.sum(violation_mask > 0))

        spatial_patterns = {
            'edge_concentrated': bool(edge_violation_ratio > 0.6),
            'has_boundary_artifacts': edge_violations['has_edge_violations'],
            'boundary_edges': edge_violations['edges_affected'],
            'boundary_severity': edge_violations['severity'],
            'linear_patterns_detected': any(v > 30 for v in edge_violations.get('linear_patterns', {}).values())
        }

        # Luma zone distribution (unless strong edge issues dominate)
        luma_distribution = {}
        if not (edge_violations['has_edge_violations'] and edge_violations['severity'] in ['medium', 'high']):
            luma_distribution = self._analyze_luma_zone_violations(violation_mask, gray)

        # Diagnostics
        diagnostics = self._generate_enhanced_diagnostic(spatial_patterns,
                                                        edge_violations,
                                                        luma_distribution)

        # Violation percentage
        violation_pixels = int(np.sum(violation_mask > 0))
        violation_percentage = (violation_pixels / (w * h)) * 100 if w * h > 0 else 0

        return {
            'spatial_patterns': spatial_patterns,
            'edge_violations': edge_violations,
            'luma_distribution': luma_distribution,
            'edge_violation_ratio': float(edge_violation_ratio),
            'diagnostics': diagnostics,
            'boundary_artifacts': edge_violations,
            'linear_pattern_info': edge_violations.get('linear_patterns', {}),
            'expansion_recommendations': edge_violations.get('expansion_recommendations', {}),
            'violation_percentage': violation_percentage
        }
    
    def _select_diverse_violations_for_thumbnails(self, violations: List[FrameViolation], 
                                                max_thumbnails: int = 5, 
                                                min_time_separation: float = 10.0) -> List[FrameViolation]:
        """
        Select violations for thumbnails ensuring temporal diversity.
        
        Args:
            violations: List of violations sorted by violation_score (highest first)
            max_thumbnails: Maximum number of thumbnails to create
            min_time_separation: Minimum time separation between selected frames (seconds)
        
        Returns:
            List of violations for thumbnail creation
        """
        if not violations:
            return []
        
        selected_violations = []
        
        # Always include the highest scoring violation
        selected_violations.append(violations[0])
        logger.info(f"  Selected thumbnail 1: Frame {violations[0].frame_num} at {violations[0].timestamp:.1f}s (score: {violations[0].violation_score:.4f})")
        
        # Select additional violations with time separation constraint
        for violation in violations[1:]:
            if len(selected_violations) >= max_thumbnails:
                break
                
            # Check if this violation is far enough from all previously selected ones
            is_far_enough = True
            for selected in selected_violations:
                time_diff = abs(violation.timestamp - selected.timestamp)
                if time_diff < min_time_separation:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                selected_violations.append(violation)
                logger.info(f"  Selected thumbnail {len(selected_violations)}: Frame {violation.frame_num} at {violation.timestamp:.1f}s (score: {violation.violation_score:.4f})")
        
        # If we couldn't find enough diverse violations, fill in with the best remaining ones
        # (but log this situation)
        if len(selected_violations) < max_thumbnails and len(violations) > len(selected_violations):
            remaining_needed = max_thumbnails - len(selected_violations)
            logger.info(f"  Only found {len(selected_violations)} violations with {min_time_separation}s separation")
            
            # Add the best remaining violations regardless of time separation
            for violation in violations:
                if violation not in selected_violations:
                    selected_violations.append(violation)
                    logger.info(f"  Added thumbnail {len(selected_violations)} (relaxed spacing): Frame {violation.frame_num} at {violation.timestamp:.1f}s")
                    remaining_needed -= 1
                    if remaining_needed <= 0:
                        break
        
        logger.info(f"  Final selection: {len(selected_violations)} thumbnails spanning {selected_violations[-1].timestamp - selected_violations[0].timestamp:.1f} seconds")
        
        return selected_violations
    
    def _generate_enhanced_diagnostic(self, spatial_patterns: Dict,
                                      edge_violations: Dict,
                                      luma_distribution: Dict) -> List[str]:
        """Human-readable diagnostics with linear pattern info."""
        diagnostics = []

        if edge_violations.get('has_edge_violations'):
            edges_str = ', '.join(edge_violations.get('edges_affected', []))
            linear_patterns = edge_violations.get('linear_patterns', {})

            high_linear = [e for e, v in linear_patterns.items() if v > 50]
            if high_linear:
                diagnostics.append(f"Linear blanking patterns on: {', '.join(high_linear)}")
            elif edge_violations.get('continuous_edges'):
                diagnostics.append(f"Continuous edge artifacts ({edges_str})")
            else:
                diagnostics.append(f"Edge artifacts ({edges_str})")

            if edge_violations.get('expansion_recommendations'):
                diagnostics.append("Border adjustment recommended")

            if edge_violations.get('severity') == 'high':
                diagnostics.append("Border detection likely missed blanking")
            elif edge_violations.get('severity') == 'medium':
                diagnostics.append("Moderate blanking detected")

        if not edge_violations.get('has_edge_violations') or edge_violations.get('severity') == 'low':
            if luma_distribution:
                primary_zone = luma_distribution.get('primary_zone')
                if primary_zone == 'highlights' and luma_distribution.get('highlight_ratio', 0) > 0.7:
                    diagnostics.append("Highlight clipping")
                elif primary_zone == 'subblack' and luma_distribution.get('subblack_ratio', 0) > 0.7:
                    diagnostics.append("Sub-black detected")

        return diagnostics if diagnostics else ["General broadcast range violations"]

    def _detect_edge_violations_enhanced(self, violation_mask, edge_width=15):
        """
        Enhanced edge violation detection that better identifies blanking patterns.
        Detects linear patterns even when pixels aren't directly adjacent.
        """
        h, w = violation_mask.shape
        edge_info = {
            'has_edge_violations': False,
            'edges_affected': [],
            'edge_percentages': {},
            'continuous_edges': [],
            'linear_patterns': {},
            'blanking_depth': {},
            'severity': 'none',
            'expansion_recommendations': {}
        }
        
        # Define edges with increased scan depth
        edges_to_check = [
            ('left', violation_mask[:, :edge_width], 'vertical'),
            ('right', violation_mask[:, -edge_width:], 'vertical'),
            ('top', violation_mask[:edge_width, :], 'horizontal'),
            ('bottom', violation_mask[-edge_width:, :], 'horizontal')
        ]
        
        for edge_name, edge_region, orientation in edges_to_check:
            if edge_region.size == 0:
                continue
            
            # Basic violation percentage
            violation_pixels = np.sum(edge_region > 0)
            total_pixels = edge_region.size
            violation_percentage = (violation_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            edge_info['edge_percentages'][edge_name] = violation_percentage
            
            # Detect linear patterns (even if not perfectly continuous)
            linear_score = 0
            if orientation == 'vertical':
                for row in range(edge_region.shape[0]):
                    row_violations = edge_region[row, :]
                    if np.any(row_violations > 0):
                        violation_positions = np.where(row_violations > 0)[0]
                        if len(violation_positions) >= 4:  
                            if edge_name == 'left' and np.max(violation_positions) <= 2:
                                linear_score += 1
                            elif edge_name == 'right' and np.min(violation_positions) >= edge_width - 3:
                                linear_score += 1
                
                linear_percentage = (linear_score / edge_region.shape[0]) * 100
                
            else:  # horizontal orientation
                # For top/bottom edges, check for vertical lines of violations
                for col in range(edge_region.shape[1]):
                    col_violations = edge_region[:, col]
                    if np.any(col_violations > 0):
                        violation_positions = np.where(col_violations > 0)[0]
                        if len(violation_positions) >= 2:
                            if edge_name == 'top' and np.max(violation_positions) <= 3:
                                linear_score += 1
                            elif edge_name == 'bottom' and np.min(violation_positions) >= edge_width - 4:
                                linear_score += 1
                
                linear_percentage = (linear_score / edge_region.shape[1]) * 100
            
            edge_info['linear_patterns'][edge_name] = linear_percentage
            
            # Determine how far blanking extends from the edge
            if violation_percentage > 15:
                max_depth = 0
                if orientation == 'vertical':
                    for row in range(edge_region.shape[0]):
                        row_violations = np.where(edge_region[row, :] > 0)[0]
                        if len(row_violations) > 0:
                            if edge_name == 'left':
                                depth = np.max(row_violations)
                            else:  # right
                                depth = edge_width - np.min(row_violations)
                            max_depth = max(max_depth, depth)
                else:  # horizontal
                    for col in range(edge_region.shape[1]):
                        col_violations = np.where(edge_region[:, col] > 0)[0]
                        if len(col_violations) > 0:
                            if edge_name == 'top':
                                depth = np.max(col_violations)
                            else:  # bottom
                                depth = edge_width - np.min(col_violations)
                            max_depth = max(max_depth, depth)
                
                edge_info['blanking_depth'][edge_name] = max_depth
            
            # Determine if this edge has significant violations
            if violation_percentage > 15 or linear_percentage > 50:  # Was 5% and 30%
                edge_info['edges_affected'].append(edge_name)
                edge_info['has_edge_violations'] = True
                
                # CHANGE 6: Higher threshold for continuous edges
                if linear_percentage > 70:  # Was 50%
                    edge_info['continuous_edges'].append(edge_name)
                
                # CHANGE 7: Less aggressive expansion
                if edge_name in edge_info['blanking_depth']:
                    recommended_expansion = edge_info['blanking_depth'][edge_name] + 2
                    edge_info['expansion_recommendations'][edge_name] = recommended_expansion
        
        # Refine severity assessment
        if len(edge_info['continuous_edges']) >= 3:  # Was 2
            edge_info['severity'] = 'high'
        elif len(edge_info['continuous_edges']) >= 2:  # Was 1
            edge_info['severity'] = 'medium'
        elif len(edge_info['edges_affected']) >= 3:  # Was 2
            edge_info['severity'] = 'low'
        elif len(edge_info['edges_affected']) >= 2 and max(edge_info['linear_patterns'].values(), default=0) > 60:  # Was 1 edge and 30%
            edge_info['severity'] = 'low'
        
        return edge_info

    def _analyze_luma_zone_violations(self, mask, gray_frame):
        """
        Analyze where violations occur in terms of brightness zones.
        """
        # Define luma zones
        subblack = (gray_frame < 64)
        midtones = (gray_frame >= 64) & (gray_frame < 192)
        highlights = (gray_frame >= 192)
        
        violations = mask > 0
        
        subblack_violations = np.sum(violations & subblack)
        midtone_violations = np.sum(violations & midtones)
        highlight_violations = np.sum(violations & highlights)
        
        total_violations = max(1, subblack_violations + midtone_violations + highlight_violations)
        
        return {
            'subblack_ratio': float(subblack_violations / total_violations),
            'midtone_ratio': float(midtone_violations / total_violations),
            'highlight_ratio': float(highlight_violations / total_violations),
            'primary_zone': self._get_primary_zone(subblack_violations, midtone_violations, highlight_violations)
        }

    def _get_primary_zone(self, subblack, midtone, highlight):
        """Determine primary brightness zone for violations"""
        zones = {'subblack': subblack, 'midtones': midtone, 'highlights': highlight}
        return max(zones, key=zones.get) if zones else 'none'


    def _analyze_aggregate_patterns(self, violations: List[FrameViolation]) -> Dict:
        """Analyze aggregate patterns across all violations with linear pattern detection"""
        if not violations:
            return {
                'requires_border_adjustment': False,
                'edge_violation_percentage': 0,
                'continuous_edge_percentage': 0,
                'linear_pattern_percentage': 0
            }
        
        # Count different types of violations
        edge_violations = 0
        continuous_edges = 0
        linear_patterns = 0
        all_affected_edges = []
        all_linear_scores = defaultdict(list)
        
        for v in violations:
            if v.pattern_analysis:
                edge_info = v.pattern_analysis.get('edge_violations', {})
                if edge_info.get('has_edge_violations'):
                    edge_violations += 1
                    all_affected_edges.extend(edge_info.get('edges_affected', []))
                if edge_info.get('continuous_edges'):
                    continuous_edges += 1
                
                # Check for linear patterns
                if v.pattern_analysis.get('spatial_patterns', {}).get('linear_patterns_detected'):
                    linear_patterns += 1
                
                # Collect linear pattern scores
                for edge, score in edge_info.get('linear_patterns', {}).items():
                    all_linear_scores[edge].append(score)
        
        edge_violation_pct = (edge_violations / len(violations)) * 100
        continuous_edge_pct = (continuous_edges / len(violations)) * 100
        linear_pattern_pct = (linear_patterns / len(violations)) * 100
        
        # Calculate average linear pattern scores
        avg_linear_patterns = {}
        for edge, scores in all_linear_scores.items():
            if scores:
                avg_linear_patterns[edge] = np.mean(scores)
        
        # Calculate expansion recommendations
        expansion_recs = {}
        unique_edges = list(set(all_affected_edges))
        for edge in unique_edges:
            edge_count = all_affected_edges.count(edge)
            if edge_count > len(violations) * 0.2:
                expansion_recs[edge] = 10 if edge_count > len(violations) * 0.5 else 5
        
        # Determine if border adjustment is needed with more nuanced logic
        requires_adjustment = (
            linear_pattern_pct > 20 or  # Strong linear patterns
            continuous_edge_pct > 15 or  # Many continuous edges
            edge_violation_pct > 30 or  # Many edge violations
            (continuous_edge_pct > 10 and len(unique_edges) >= 2) or  # Multiple edges affected
            any(score > 40 for score in avg_linear_patterns.values())  # High linear scores
        )
        
        return {
            'requires_border_adjustment': requires_adjustment,
            'edge_violation_percentage': edge_violation_pct,
            'continuous_edge_percentage': continuous_edge_pct,
            'linear_pattern_percentage': linear_pattern_pct,
            'linear_pattern_scores': avg_linear_patterns,
            'boundary_edges_detected': unique_edges,
            'expansion_recommendations': expansion_recs
        }
    
    def _generate_actionable_report(self, violations: List[FrameViolation],
                               aggregate_patterns: Dict) -> Dict:
        """Generate actionable recommendations based on violation patterns"""
        if not violations:
            return {
                'overall_assessment': 'No BRNG violations detected',
                'action_priority': 'none',
                'recommendations': []
            }
        
        recommendations = []
        
        # Check for border adjustment needs with more detailed analysis
        if aggregate_patterns['requires_border_adjustment']:
            edges = aggregate_patterns.get('boundary_edges_detected', [])
            linear_pct = aggregate_patterns.get('linear_pattern_percentage', 0)
            continuous_pct = aggregate_patterns.get('continuous_edge_percentage', 0)
            linear_scores = aggregate_patterns.get('linear_pattern_scores', {})
            
            # Determine severity based on linear patterns
            max_linear_score = max(linear_scores.values()) if linear_scores else 0
            
            if max_linear_score > 50 or linear_pct > 40:
                severity = 'high'
                description = f'Strong linear blanking patterns detected ({linear_pct:.1f}% of frames). Border detection likely missed blanking lines.'
            elif linear_pct > 20 or continuous_pct > 25:
                severity = 'medium'
                description = f'Moderate blanking patterns detected. {linear_pct:.1f}% frames show linear patterns, {continuous_pct:.1f}% have continuous edges.'
            else:
                severity = 'low'
                description = f'Edge violations detected on {", ".join(edges)}. May indicate minor border detection issues.'
            
            recommendations.append({
                'issue': 'Border Detection Needs Adjustment',
                'severity': severity,
                'action': f'Re-run border detection with adjusted parameters',
                'affected_edges': edges,
                'linear_pattern_scores': linear_scores,
                'description': description
            })
            action_priority = 'high' if severity == 'high' else 'medium'
        else:
            # Check for content violations
            avg_violation_pct = np.mean([v.violation_percentage for v in violations])
            max_violation_pct = max([v.violation_percentage for v in violations])
            
            if avg_violation_pct > 1.0 or max_violation_pct > 5.0:
                recommendations.append({
                    'issue': 'Content BRNG Violations',
                    'severity': 'medium',
                    'action': 'Review source material or encoding parameters',
                    'description': f'Average violation: {avg_violation_pct:.2f}%, Maximum: {max_violation_pct:.2f}%'
                })
                action_priority = 'medium'
            elif avg_violation_pct > 0.1:
                recommendations.append({
                    'issue': 'Minor BRNG Violations',
                    'severity': 'low',
                    'action': 'Review may be needed for broadcast compliance',
                    'description': f'Low-level violations detected: {avg_violation_pct:.2f}% average'
                })
                action_priority = 'low'
            else:
                action_priority = 'none'
        
        return {
            'overall_assessment': self._get_overall_assessment(violations, aggregate_patterns),
            'action_priority': action_priority,
            'recommendations': recommendations,
            'summary_statistics': {
                'total_violations': len(violations),
                'average_violation_percentage': np.mean([v.violation_percentage for v in violations]) if violations else 0,
                'max_violation_percentage': max([v.violation_percentage for v in violations]) if violations else 0,
                'edge_violation_percentage': aggregate_patterns.get('edge_violation_percentage', 0),
                'linear_pattern_percentage': aggregate_patterns.get('linear_pattern_percentage', 0)
            }
        }

    def _get_overall_assessment(self, violations: List[FrameViolation],
                            aggregate_patterns: Dict) -> str:
        """Generate overall assessment text with enhanced pattern recognition"""
        if not violations:
            return "Video is broadcast-safe with no BRNG violations"
        
        if aggregate_patterns['requires_border_adjustment']:
            linear_pct = aggregate_patterns.get('linear_pattern_percentage', 0)
            linear_scores = aggregate_patterns.get('linear_pattern_scores', {})
            max_linear = max(linear_scores.values()) if linear_scores else 0
            
            if max_linear > 50 or linear_pct > 40:
                return "Border detection missed blanking lines - strong linear patterns at frame edges"
            elif linear_pct > 20:
                return "Border adjustment recommended - moderate blanking patterns detected"
            else:
                return "Border detection adjustment required - edges contain violations"
        
        avg_violation_pct = np.mean([v.violation_percentage for v in violations])
        if avg_violation_pct < 0.01:
            return "Video appears broadcast-compliant with minimal violations"
        elif avg_violation_pct < 0.1:
            return "Minor broadcast range issues detected - likely acceptable"
        elif avg_violation_pct < 1.0:
            return "Moderate broadcast range issues detected"
        else:
            return "Significant broadcast range issues requiring attention"
    
    def _create_diagnostic_thumbnails(self, violations: List[FrameViolation],
                             temp_video_paths: List[Dict],
                             output_dir: Path) -> List[str]:
        """Create 4-quadrant diagnostic thumbnails with enhanced violation visibility"""
        thumbnails = []
        
        for i, violation in enumerate(violations[:5]):
            # Find which video contains this violation
            video_info = None
            for vinfo in temp_video_paths:
                if vinfo['start_time'] <= violation.timestamp < vinfo['start_time'] + vinfo['duration']:
                    video_info = vinfo
                    break
            
            if not video_info:
                logger.warning(f"Could not find video segment for violation at {violation.timestamp}s")
                continue
            
            # Open the videos
            cap_h = cv2.VideoCapture(str(video_info['highlighted']))
            cap_o = cv2.VideoCapture(str(video_info['original']))
            
            if not cap_h.isOpened() or not cap_o.isOpened():
                logger.warning(f"Failed to open videos for thumbnail")
                cap_h.release()
                cap_o.release()
                continue
            
            fps = cap_h.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame position relative to this video segment
            relative_time = violation.timestamp - video_info['start_time']
            frame_idx = int(relative_time * fps)
            
            cap_h.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap_o.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret_h, frame_h = cap_h.read()
            ret_o, frame_o = cap_o.read()
            
            if not ret_h or not ret_o:
                cap_h.release()
                cap_o.release()
                continue
            
            # Create 4-quadrant visualization
            h, w = frame_h.shape[:2]
            padding = 10
            viz_height = h * 2 + padding
            viz_width = w * 2 + padding
            viz = np.full((viz_height, viz_width, 3), (64, 64, 64), dtype=np.uint8)
            
            # Top-left: Original
            viz[0:h, 0:w] = frame_o
            
            # Top-right: Highlighted
            viz[0:h, w+padding:w*2+padding] = frame_h
            
            # Bottom-left: Enhanced violations visualization
            violations_only = self._create_enhanced_violations_display(frame_h, frame_o)
            viz[h+padding:h*2+padding, 0:w] = violations_only
            
            # Bottom-right: Analysis info
            info_panel = np.zeros((h, w, 3), dtype=np.uint8)
            self._add_info_text(info_panel, violation)
            viz[h+padding:h*2+padding, w+padding:w*2+padding] = info_panel
            
            # Add labels with better visibility
            self._add_quadrant_labels(viz, w, h, padding)
            
            # Save thumbnail
            thumb_filename = f"{self.video_path.stem}_brng_{i:03d}_{violation.timestamp:.1f}s.jpg"
            thumb_path = output_dir / thumb_filename
            cv2.imwrite(str(thumb_path), viz)
            thumbnails.append(str(thumb_path))
            
            cap_h.release()
            cap_o.release()
        
        return thumbnails

    def _create_enhanced_violations_display(self, frame_h: np.ndarray, frame_o: np.ndarray) -> np.ndarray:
        """Create enhanced violations display with multiple visualization techniques"""
        
        # Method 1: Use sensitive detection for more pixels
        violation_mask_sensitive = self._detect_differential_violations_frame(
            frame_h, frame_o, sensitivity='visualization'
        )
        
        # Method 2: Create a simple difference mask as backup
        diff_mask = self._create_simple_difference_mask(frame_h, frame_o)
        
        # Method 3: Combine both masks
        combined_mask = cv2.bitwise_or(violation_mask_sensitive, diff_mask)
        
        # Create the visualization
        violations_only = np.zeros_like(frame_o)
        
        # Use bright cyan for primary violations
        violations_only[violation_mask_sensitive > 0] = [0, 255, 255]  # Cyan
        
        # Use yellow for additional differences (less confident but visible)
        additional_violations = cv2.bitwise_and(diff_mask, cv2.bitwise_not(violation_mask_sensitive))
        violations_only[additional_violations > 0] = [0, 255, 255]  # Keep cyan but with lower intensity
        
        # Enhance visibility by adding slight transparency overlay where violations occur
        violation_overlay = np.zeros_like(frame_o)
        all_violations = combined_mask > 0
        
        if np.any(all_violations):
            # Add the original frame content at low opacity where violations occur
            violation_overlay[all_violations] = frame_o[all_violations] * 0.3
            violations_only = cv2.addWeighted(violations_only, 0.8, violation_overlay, 0.2, 0)
        
        return violations_only

    def _create_simple_difference_mask(self, frame_h: np.ndarray, frame_o: np.ndarray) -> np.ndarray:
        """Create a simple difference mask as backup for visualization"""
        
        # Convert to grayscale for difference calculation
        gray_h = cv2.cvtColor(frame_h, cv2.COLOR_BGR2GRAY)
        gray_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray_h, gray_o)
        
        # Threshold for any noticeable change
        _, diff_mask = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
        
        # Remove very small differences (noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Additional filtering for magenta-like colors
        h_b, h_g, h_r = cv2.split(frame_h)
        o_b, o_g, o_r = cv2.split(frame_o)
        
        # Look for any increase in red or blue channels
        red_increase = (h_r.astype(np.float32) - o_r.astype(np.float32)) > 3
        blue_increase = (h_b.astype(np.float32) - o_b.astype(np.float32)) > 3
        
        # Combine with difference mask
        color_change = (red_increase | blue_increase).astype(np.uint8) * 255
        final_mask = cv2.bitwise_and(diff_mask, color_change)
        
        return final_mask

    def _add_quadrant_labels(self, viz: np.ndarray, w: int, h: int, padding: int):
        """Add clear labels to each quadrant"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        thickness = 2
        
        labels = [
            ("Original", (10, 25)),
            ("BRNG Highlighted", (w + padding + 10, 25)),
            ("Violations Only", (10, h + padding + 25)),
            ("Analysis Data", (w + padding + 10, h + padding + 25))
        ]
        
        for text, (x, y) in labels:
            # Get text size for background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(viz, (x - 5, y - text_h - 10), 
                        (x + text_w + 5, y + baseline), bg_color, -1)
            
            # Draw text
            cv2.putText(viz, text, (x, y - 5), font, font_scale, text_color, thickness)

    def _add_info_text(self, panel: np.ndarray, violation: FrameViolation):
        """Add analysis info text to panel"""
        h, w = panel.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        lines = [
            f"Frame: {violation.frame_num}",
            f"Time: {violation.timestamp:.2f}s",
            f"BRNG: {violation.brng_value:.4f}%",
            f"Violations: {violation.violation_percentage:.4f}%",
            f"Pixels: {violation.violation_pixels}"
        ]
        
        if violation.diagnostics:
            lines.append(f"Issues: {', '.join(violation.diagnostics[:2])}")
        
        y = 30
        for line in lines:
            cv2.putText(panel, line, (10, y), font, 0.5, (255, 255, 255), 1)
            y += 25


class IntegratedSignalstatsAnalyzer:
    """Signalstats analyzer with QCTools integration"""
    
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.qctools_report = self._find_qctools_report()
        self._init_video_properties()
        
    def _init_video_properties(self):
        """Initialize video properties"""
        cap = cv2.VideoCapture(self.video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()
    
    def _find_qctools_report(self) -> Optional[str]:
        """Find existing QCTools report"""
        video_path = Path(self.video_path)
        video_id = video_path.stem
        
        search_patterns = [
            video_path.parent / f"{video_id}.qctools.xml.gz",
            video_path.parent / f"{video_id}_qc_metadata" / f"{video_id}.qctools.xml.gz",
            video_path.parent / f"{video_id}_vrecord_metadata" / f"{video_id}.qctools.xml.gz",
        ]
        
        for pattern in search_patterns:
            if pattern.exists():
                return str(pattern)
        
        return None
    
    def analyze_with_signalstats(self,
                            border_data: BorderDetectionResult = None,
                            content_start_time: float = 0,
                            color_bars_end_time: float = None,
                            analysis_duration: int = 60,
                            num_periods: int = 3) -> SignalstatsResult:
        """
        Analyze using signalstats, comparing full frame (QCTools) vs active area (FFprobe).
        """
        # Determine analysis periods
        analysis_periods = self._find_analysis_periods(
            content_start_time, 
            color_bars_end_time,
            analysis_duration,
            num_periods,
            border_data.quality_frame_hints if border_data else None
        )
        
        # Log analysis configuration
        logger.info(f"  Running {len(analysis_periods)} analysis periods:")
        for i, (start_time, duration) in enumerate(analysis_periods):
            end_time = start_time + duration
            start_tc = self._seconds_to_timecode(start_time)
            end_tc = self._seconds_to_timecode(end_time)
            logger.info(f"    Period {i+1}: {start_tc} - {end_tc} ({duration}s)")
        
        # Log active area vs full frame comparison
        active_area = border_data.active_area if border_data else None
        if active_area:
            x, y, w, h = active_area
            full_w, full_h = self.width, self.height
            crop_pct = (w * h) / (full_w * full_h) * 100
            logger.info(f"  Comparison mode: Full frame ({full_w}x{full_h}) vs Active area ({w}x{h} at {x},{y})")
            logger.info(f"  Active area is {crop_pct:.1f}% of full frame")
        else:
            logger.info(f"  Comparison mode: Full frame analysis only (no border detection)")
        
        # Analyze periods
        all_results = []
        used_qctools = False
        comparison_results = []
        
        for i, (start_time, duration) in enumerate(analysis_periods):
            logger.info(f"  Analyzing period {i+1} ({self._seconds_to_timecode(start_time)} - {self._seconds_to_timecode(start_time + duration)}):")
            
            period_comparison = {
                'period': i+1,
                'time_range': (start_time, start_time + duration)
            }
            
            # Get QCTools data (full frame)
            qctools_result = None
            if self.qctools_report:
                qctools_result = self._parse_qctools_brng_period(start_time, start_time + duration, i+1)
                if qctools_result:
                    used_qctools = True
                    period_comparison['qctools_full_frame'] = {
                        'violations_pct': (qctools_result['frames_with_violations'] / 
                                        qctools_result['frames_analyzed'] * 100) if qctools_result['frames_analyzed'] > 0 else 0,
                        'max_brng': max(qctools_result['brng_values']) * 100 if qctools_result['brng_values'] else 0
                    }
                    logger.info(f"    QCTools (full frame): {period_comparison['qctools_full_frame']['violations_pct']:.1f}% violations, "
                            f"max BRNG: {period_comparison['qctools_full_frame']['max_brng']:.4f}%")
            
            # Get FFprobe data (active area only)
            if active_area:
                logger.info(f"    Running FFprobe on active area only...")
                ffprobe_result = self._analyze_with_ffprobe_period(
                    active_area, start_time, duration, i+1
                )
                if ffprobe_result:
                    period_comparison['ffprobe_active_area'] = {
                        'violations_pct': (ffprobe_result['frames_with_violations'] / 
                                        ffprobe_result['frames_analyzed'] * 100) if ffprobe_result['frames_analyzed'] > 0 else 0,
                        'max_brng': max(ffprobe_result['brng_values']) * 100 if ffprobe_result['brng_values'] else 0
                    }
                    logger.info(f"    FFprobe (active area): {period_comparison['ffprobe_active_area']['violations_pct']:.1f}% violations, "
                            f"max BRNG: {period_comparison['ffprobe_active_area']['max_brng']:.4f}%")
                    
                    # Compare results
                    if qctools_result and period_comparison.get('qctools_full_frame'):
                        full_violations = period_comparison['qctools_full_frame']['violations_pct']
                        active_violations = period_comparison['ffprobe_active_area']['violations_pct']
                        
                        if full_violations > active_violations + 5:
                            logger.info(f"    → Border violations detected: Full frame has {full_violations - active_violations:.1f}% more violations")
                            period_comparison['diagnosis'] = 'border_violations'
                        elif active_violations > 10:
                            logger.info(f"    → Content violations: Active area itself has {active_violations:.1f}% violations")
                            period_comparison['diagnosis'] = 'content_violations'
                        else:
                            logger.info(f"    → Minimal violations in both full frame and active area")
                            period_comparison['diagnosis'] = 'minimal_violations'
                    
                    # Use FFprobe result for aggregate
                    all_results.append(ffprobe_result)
                else:
                    # Fall back to QCTools if FFprobe fails
                    if qctools_result:
                        all_results.append(qctools_result)
            else:
                # No active area defined, use QCTools or FFprobe on full frame
                if qctools_result:
                    all_results.append(qctools_result)
                else:
                    logger.info(f"    Using FFprobe on full frame (no border detection)")
                    ffprobe_result = self._analyze_with_ffprobe_period(
                        None, start_time, duration, i+1
                    )
                    if ffprobe_result:
                        all_results.append(ffprobe_result)
            
            comparison_results.append(period_comparison)
        
        # Aggregate results
        if not all_results:
            return SignalstatsResult(
                violation_percentage=0,
                max_brng=0,
                avg_brng=0,
                analysis_periods=analysis_periods,
                diagnosis="No data available",
                used_qctools=False
            )
        
        # Calculate aggregates
        total_violations = sum(r.get('frames_with_violations', 0) for r in all_results)
        total_frames = sum(r.get('frames_analyzed', 0) for r in all_results)
        all_brng_values = []
        for r in all_results:
            all_brng_values.extend(r.get('brng_values', []))
        
        violation_pct = (total_violations / total_frames * 100) if total_frames > 0 else 0
        max_brng = max(all_brng_values) * 100 if all_brng_values else 0
        avg_brng = np.mean(all_brng_values) * 100 if all_brng_values else 0
        
        # Generate comprehensive diagnosis
        diagnosis = self._generate_comprehensive_diagnosis(
            violation_pct, max_brng, comparison_results, active_area is not None
        )
        
        # Log final comparison summary
        logger.info(f"\n  === Signalstats Analysis Summary ===")
        logger.info(f"  Active area results (what matters for QC):")
        logger.info(f"    Frames with violations: {total_violations:,} / {total_frames:,} ({violation_pct:.1f}%)")
        logger.info(f"    Max BRNG value: {max_brng:.4f}%")
        logger.info(f"    Average BRNG value: {avg_brng:.4f}%")
        logger.info(f"  Diagnosis: {diagnosis}")
        
        return SignalstatsResult(
            violation_percentage=violation_pct,
            max_brng=max_brng,
            avg_brng=avg_brng,
            analysis_periods=analysis_periods,
            diagnosis=diagnosis,
            used_qctools=used_qctools
        )

    def _generate_comprehensive_diagnosis(self, violation_pct: float, max_brng: float, 
                                        comparison_results: List[Dict], has_active_area: bool) -> str:
        """Generate diagnosis based on full frame vs active area comparison"""
        
        if not has_active_area:
            # No border detection, standard diagnosis
            if violation_pct < 10 and max_brng < 0.1:
                return "Video appears broadcast-compliant"
            elif violation_pct < 50 and max_brng < 1.0:
                return "Minor BRNG violations - likely acceptable"
            elif max_brng > 2.0:
                return "Significant BRNG violations requiring correction"
            else:
                return "Moderate BRNG violations detected"
        
        # Analyze comparison results
        border_violation_periods = sum(1 for r in comparison_results if r.get('diagnosis') == 'border_violations')
        content_violation_periods = sum(1 for r in comparison_results if r.get('diagnosis') == 'content_violations')
        
        if border_violation_periods > content_violation_periods:
            return "BRNG violations primarily in border areas - active content appears broadcast-safe"
        elif content_violation_periods > 0:
            if violation_pct > 50:
                return "Significant BRNG violations in active picture area - requires correction"
            else:
                return "BRNG violations detected in active picture area - review recommended"
        else:
            return "Video appears broadcast-compliant with properly detected borders"
    
    def _seconds_to_timecode(self, seconds: float) -> str:
        """Convert seconds to MM:SS.mmm format"""
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:06.3f}"
    
    def _find_analysis_periods(self, content_start: float, color_bars_end: float,
                          duration: int, num_periods: int,
                          quality_hints: List[Tuple[float, float]] = None) -> List[Tuple[float, int]]:
        """Find good analysis periods, using quality hints if available"""
        # Start after color bars with a safety margin
        effective_start = max(content_start, color_bars_end or 0) + 10
        
        # Log what we're doing
        logger.info(f"  Content starts at {effective_start:.1f}s (after color bars at {color_bars_end:.1f}s)")
        
        # Use quality hints if available (but ensure they're after color bars)
        if quality_hints:
            valid_hints = [(t, q) for t, q in quality_hints if t >= effective_start]
            if len(valid_hints) >= num_periods:
                periods = []
                for time_hint, _ in valid_hints[:num_periods]:
                    period_start = max(effective_start, time_hint - duration/2)
                    if period_start + duration <= self.duration - 30:
                        periods.append((period_start, duration))
                if len(periods) >= num_periods:
                    return periods
        
        # Fall back to even distribution
        available_duration = self.duration - effective_start - 30
        if available_duration < duration * num_periods:
            # If not enough space for all periods, reduce number or duration
            if available_duration >= duration:
                # At least one full period possible
                periods = []
                actual_periods = min(num_periods, int(available_duration / duration))
                spacing = (available_duration - duration) / max(1, actual_periods - 1)
                for i in range(actual_periods):
                    start = effective_start + i * spacing
                    periods.append((start, duration))
                return periods
            else:
                # Reduce duration
                return [(effective_start, min(duration, available_duration))]
        
        periods = []
        spacing = (available_duration - duration) / max(1, num_periods - 1)
        for i in range(num_periods):
            start = effective_start + i * spacing
            periods.append((start, duration))
        
        return periods
    
    def _parse_qctools_brng_period(self, start_time: float, end_time: float, period_num: int) -> Optional[Dict]:
        """Parse BRNG values from QCTools report for specific time period"""
        if not self.qctools_report:
            return None
        
        # Create a parser instance for this specific period
        parser = QCToolsParser(self.qctools_report, self.fps)
        
        # Parse violations for the specific time range
        violations = parser.parse_for_violations_streaming_period(
            start_time=start_time,
            end_time=end_time,
            period_num=period_num,
            max_frames=1000
        )
        
        if not violations:
            logger.info(f"    No violations found in period {period_num}")
            return None
        
        return {
            'frames_analyzed': len(violations),
            'frames_with_violations': len([v for v in violations if v.violation_score > 0]),
            'brng_values': [v.violation_score for v in violations],
            'source': 'qctools',
            'period_num': period_num,
            'time_range': (start_time, end_time)
        }
    
    def _should_use_qctools(self, qctools_result: Dict) -> bool:
        """Decide if QCTools data is sufficient"""
        if not qctools_result:
            return False
        
        violation_pct = (qctools_result['frames_with_violations'] / 
                        qctools_result['frames_analyzed'] * 100)
        max_brng = max(qctools_result['brng_values']) if qctools_result['brng_values'] else 0
        
        # Log the QCTools results for this period
        period_num = qctools_result.get('period_num', '?')
        logger.info(f"    Period {period_num} QCTools results: {qctools_result['frames_analyzed']:,} frames, "
                   f"{qctools_result['frames_with_violations']:,} violations ({violation_pct:.1f}%), "
                   f"max BRNG: {max_brng*100:.4f}%")
        
        # Use QCTools if violations are minimal or if we have good data
        return True  # Always use QCTools data when available for consistency
    
    def _analyze_with_ffprobe_period(self, active_area: Tuple, 
                                   start_time: float, duration: int, period_num: int) -> Dict:
        """Analyze using FFprobe signalstats for specific period"""
        # Build filter chain
        filter_chain = f"movie={shlex.quote(self.video_path)}"
        filter_chain += f",select='between(t\\,{start_time}\\,{start_time + duration})'"
        
        if active_area:
            x, y, w, h = active_area
            filter_chain += f",crop={w}:{h}:{x}:{y}"
        
        filter_chain += ",signalstats=stat=brng"
        
        cmd = [
            'ffprobe',
            '-f', 'lavfi',
            '-i', filter_chain,
            '-show_entries', 'frame_tags=lavfi.signalstats.BRNG',
            '-of', 'csv=p=0'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"    FFprobe failed for period {period_num}")
                return None
            
            # Parse output
            brng_values = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        brng_values.append(float(line.strip()))
                    except:
                        pass
            
            frames_with_violations = len([v for v in brng_values if v > 0])
            violation_pct = (frames_with_violations / len(brng_values) * 100) if brng_values else 0
            max_brng = max(brng_values) if brng_values else 0
            
            logger.info(f"    Period {period_num} FFprobe results: {len(brng_values):,} frames, "
                       f"{frames_with_violations:,} violations ({violation_pct:.1f}%), "
                       f"max BRNG: {max_brng*100:.4f}%")
            
            return {
                'frames_analyzed': len(brng_values),
                'frames_with_violations': frames_with_violations,
                'brng_values': brng_values,
                'source': 'ffprobe',
                'period_num': period_num
            }
        except Exception as e:
            logger.error(f"FFprobe error for period {period_num}: {e}")
            return None


class EnhancedFrameAnalysis:
    """
    Main coordinator for enhanced frame analysis.
    Combines efficiency of refactored version with sophistication of original.
    """
    
    def __init__(self, video_path: str, output_dir: str = None):
        self.video_path = Path(video_path)
        self.video_id = self.video_path.stem
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent
        self.output_dir.mkdir(exist_ok=True)

        # Store config manager as an instance attribute
        self.config_mgr = ConfigManager()
        self.config_mgr.refresh_configs()
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        
        # Initialize components
        self.border_detector = SophisticatedBorderDetector(video_path)
        self.brng_analyzer = None  # Will be initialized with border data
        self.signalstats_analyzer = IntegratedSignalstatsAnalyzer(video_path)
        self.signalstats_analyzer = IntegratedSignalstatsAnalyzer(video_path)

        
        # Find QCTools report
        self.qctools_report = self._find_qctools_report()
        self.qctools_parser = None
        if self.qctools_report:
            self.qctools_parser = QCToolsParser(self.qctools_report)
    
    def _find_qctools_report(self) -> Optional[str]:
        """Find QCTools report for the video"""
        search_paths = [
            self.video_path.parent / f"{self.video_id}.qctools.xml.gz",
            self.video_path.parent / f"{self.video_id}_qc_metadata" / f"{self.video_id}.qctools.xml.gz",
            self.video_path.parent / f"{self.video_id}_vrecord_metadata" / f"{self.video_id}.qctools.xml.gz",
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found QCTools report: {path}")
                return str(path)
        
        return None
    
    def analyze(self, 
               method: str = 'sophisticated',
               duration_limit: int = 300,
               skip_color_bars: bool = True,
               max_refinement_iterations: int = 3) -> Dict:
        """
        Run complete enhanced frame analysis with optional iterative refinement.
        
        Args:
            method: 'sophisticated' or 'simple' border detection
            duration_limit: Maximum duration to analyze (seconds)
            skip_color_bars: Whether to skip color bars at start
            max_refinement_iterations: Maximum border refinement iterations
        
        Returns:
            Complete analysis results dictionary
        """
        results = {
            'video_path': str(self.video_path),
            'video_id': self.video_id,
            'analysis_method': 'enhanced',
            'qctools_report_available': self.qctools_report is not None
        }

        frame_config = self.checks_config.outputs.frame_analysis
        
        # Step 1: Detect color bars
        color_bars_end_time = 0
        if skip_color_bars:
            color_bars_end_time = self._detect_color_bars_duration()
            if color_bars_end_time > 0:
                logger.info(f"Color bars detected, ending at {color_bars_end_time:.1f}s")
                results['color_bars_end_time'] = color_bars_end_time

        # Step 2: Parse QCTools for initial violations
        violations = []
        if self.qctools_parser:
            logger.info("Parsing QCTools report for violations...")
            violations = self.qctools_parser.parse_for_violations_streaming(
                max_frames=100,
                skip_color_bars=skip_color_bars,
                color_bars_end_time=color_bars_end_time
            )
            frames_with_qctools_violations = len(violations)
            
            if frames_with_qctools_violations == 0:
                results['qctools_violations_found'] = "No BRNG violations detected in content"
            else:
                results['qctools_violations_found'] = frames_with_qctools_violations
        else:
            logger.info("No QCTools report found, proceeding with direct analysis")
        
        # Step 3: Initial border detection
        logger.info(f"Detecting borders using {method} method...")
        border_results = self.border_detector.detect_borders_with_quality_assessment(
            violations=violations,
            method=method
        )
        results['initial_borders'] = asdict(border_results)
        
        # Step 4: Signalstats analysis (MOVED UP from Step 5)
        logger.info("Running signalstats analysis to identify key analysis periods...")
        signalstats_results = self.signalstats_analyzer.analyze_with_signalstats(
            border_data=border_results,
            content_start_time=0,
            color_bars_end_time=color_bars_end_time,
            analysis_duration=frame_config.signalstats_duration,
            num_periods=frame_config.signalstats_periods
        )
        results['signalstats'] = asdict(signalstats_results)
        
        # Extract the analysis periods from signalstats results
        analysis_periods = signalstats_results.analysis_periods
        logger.info(f"Identified {len(analysis_periods)} key periods for BRNG analysis")
        
        # Step 5: BRNG analysis using signalstats periods
        logger.info("Analyzing BRNG violations in identified periods...")
        self.brng_analyzer = DifferentialBRNGAnalyzer(self.video_path, border_results)
        
        brng_results = self.brng_analyzer.analyze_with_differential_detection(
            output_dir=self.output_dir,
            duration_limit=duration_limit,
            skip_start_seconds=color_bars_end_time,
            qctools_violations=violations,
            analysis_periods=analysis_periods  # Pass the periods from signalstats
        )
        results['brng_analysis'] = asdict(brng_results) if brng_results else None
        
        # Step 6: Iterative border refinement (if needed)
        refinement_iterations = 0
        if method == 'sophisticated' and brng_results and brng_results.requires_border_adjustment:
            logger.info("Border refinement needed - detected BRNG violations at frame edges")
            
            edge_pct = brng_results.aggregate_patterns.get('edge_violation_percentage', 0)
            continuous_pct = brng_results.aggregate_patterns.get('continuous_edge_percentage', 0)
            logger.info(f"  Edge violations: {edge_pct:.1f}% of analyzed frames")
            logger.info(f"  Continuous edge patterns: {continuous_pct:.1f}% of analyzed frames")
            
            while (refinement_iterations < max_refinement_iterations and 
                brng_results.requires_border_adjustment):
                
                refinement_iterations += 1
                logger.info(f"Refinement iteration {refinement_iterations}/{max_refinement_iterations}:")
                
                # Refine borders
                previous_area = border_results.active_area
                border_results = self.border_detector.refine_borders(border_results, brng_results)
                new_area = border_results.active_area
                
                logger.info(f"  Active area: {previous_area[2]}x{previous_area[3]} → {new_area[2]}x{new_area[3]}")
                
                # Re-run signalstats with new borders to get updated periods
                logger.info("  Re-running signalstats with refined borders...")
                signalstats_results = self.signalstats_analyzer.analyze_with_signalstats(
                    border_data=border_results,
                    content_start_time=0,
                    color_bars_end_time=color_bars_end_time,
                    analysis_duration=frame_config.signalstats_duration,
                    num_periods=frame_config.signalstats_periods
                )
                analysis_periods = signalstats_results.analysis_periods
                
                # Re-analyze BRNG with new borders and periods
                self.brng_analyzer = DifferentialBRNGAnalyzer(self.video_path, border_results)
                previous_brng = brng_results
                
                brng_results = self.brng_analyzer.analyze_with_differential_detection(
                    output_dir=self.output_dir,
                    duration_limit=duration_limit,
                    skip_start_seconds=color_bars_end_time,
                    qctools_violations=violations,
                    analysis_periods=analysis_periods  # Use updated periods
                )
                
                # Check for improvement
                improved = self._is_meaningful_improvement(previous_brng, brng_results)
                if not improved:
                    logger.info("  Stopping refinement - further adjustments unlikely to help")
                    break
            
            results['refinement_iterations'] = refinement_iterations
            results['final_borders'] = asdict(border_results)
            results['final_brng_analysis'] = asdict(brng_results) if brng_results else None
            results['final_signalstats'] = asdict(signalstats_results)
        
        # Step 7: Generate comprehensive summary
        results['summary'] = self._generate_summary(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _detect_color_bars_duration(self) -> float:
        """Detect color bars duration from video or QCTools report"""
        # First check if qct-parse already detected color bars
        report_dir = self.video_path.parent / f"{self.video_id}_report_csvs"
        colorbars_csv = report_dir / "qct-parse_colorbars_durations.csv"
        
        if colorbars_csv.exists():
            import csv
            try:
                with open(colorbars_csv, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2 and "color bars found" in rows[0][0]:
                        # Parse end timestamp
                        end_str = rows[1][1] if len(rows[1]) > 1 else None
                        if end_str:
                            # Convert timestamp to seconds
                            parts = end_str.split(':')
                            if len(parts) == 3:
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                seconds = float(parts[2])
                                return hours * 3600 + minutes * 60 + seconds
            except Exception as e:
                logger.warning(f"Could not parse color bars CSV: {e}")
        
        # If no qct-parse data, could implement direct detection here
        # For now, return 0 to indicate no color bars detected
        return 0
    
    def _is_meaningful_improvement(self, previous: BRNGAnalysisResult, 
                                  current: BRNGAnalysisResult) -> bool:
        """Check if refinement produced meaningful improvement"""
        if not previous or not current:
            return False
        
        prev_violations = len(previous.violations)
        curr_violations = len(current.violations)
        
        if curr_violations < prev_violations * 0.8:  # 20% improvement
            return True
        
        prev_worst = previous.violations[0].violation_percentage if previous.violations else 0
        curr_worst = current.violations[0].violation_percentage if current.violations else 0
        
        if curr_worst < prev_worst * 0.8:  # 20% improvement in worst case
            return True
        
        return False
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate comprehensive human-readable summary"""
        lines = []
        lines.append(f"Enhanced Frame Analysis Summary - {self.video_id}")
        lines.append("=" * 60)
        
        # QCTools status
        if results['qctools_report_available']:
            lines.append(f"✓ QCTools report found")
            if 'qctools_violations_found' in results:
                violations = results['qctools_violations_found']
                if isinstance(violations, str):
                    lines.append(f"  {violations}")
                else:
                    lines.append(f"  Frames with BRNG violations: {violations}")
        
        # Border detection
        if 'initial_borders' in results:
            borders = results.get('final_borders', results['initial_borders'])
            if borders['active_area']:
                x, y, w, h = borders['active_area']
                lines.append(f"\nActive area: {w}x{h} at ({x},{y})")
                lines.append(f"Detection method: {borders['detection_method']}")
        
        # BRNG analysis
        if results.get('brng_analysis'):
            brng = results.get('final_brng_analysis', results['brng_analysis'])
            report = brng['actionable_report']
            lines.append(f"\nBRNG Assessment: {report['overall_assessment']}")
            lines.append(f"Priority: {report['action_priority'].upper()}")
        
        # Signalstats
        if results.get('signalstats'):
            stats = results['signalstats']
            lines.append(f"\nSignalstats: {stats['diagnosis']}")
            lines.append(f"  Violations: {stats['violation_percentage']:.1f}% of frames")
            lines.append(f"  Max BRNG: {stats['max_brng']:.4f}%")
        
        # Refinement info
        if 'refinement_iterations' in results:
            lines.append(f"\nBorder refinement: {results['refinement_iterations']} iteration(s)")
        
        return "\n".join(lines)
    
    def _save_results(self, results: Dict):
        """Save analysis results to JSON"""
        import copy
        output_file = self.output_dir / f"{self.video_id}_enhanced_frame_analysis.json"
        
        # Deep copy to avoid modifying the original
        results_copy = copy.deepcopy(results)
        
        # Convert any remaining non-serializable objects
        def clean_for_json(obj, seen=None):
            if seen is None:
                seen = set()
            
            # Check for circular reference
            obj_id = id(obj)
            if obj_id in seen:
                return None  # or return a placeholder like "<circular reference>"
            
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            
            seen.add(obj_id)
            
            if isinstance(obj, dict):
                return {k: clean_for_json(v, seen.copy()) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item, seen.copy()) for item in obj]
            elif hasattr(obj, '__dict__'):
                return clean_for_json(obj.__dict__, seen.copy())
            else:
                return str(obj)
        
        cleaned_results = clean_for_json(results_copy)
        
        with open(output_file, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        print(results['summary'])


# Public API function for processing_mgmt to call
def analyze_frame_quality(video_path: str, 
                         border_data_path: str = None,
                         output_dir: str = None,
                         frame_config: 'FrameAnalysisConfig' = None,
                         color_bars_end_time: float = None) -> Dict:
    """
    Main entry point for frame analysis from processing_mgmt.
    
    Args:
        video_path: Path to video file
        border_data_path: Optional path to existing border data JSON
        output_dir: Output directory for results
        frame_config: FrameAnalysisConfig dataclass with analysis parameters
        color_bars_end_time: End time of color bars if detected
    
    Returns:
        Complete analysis results dictionary
    """
    # Use config dataclass directly
    if frame_config is None:
        # Use defaults if no config provided
        from AV_Spex.utils.config_setup import FrameAnalysisConfig
        frame_config = FrameAnalysisConfig()
    
    # Extract parameters directly from dataclass
    method = frame_config.border_detection_mode
    duration_limit = frame_config.brng_duration_limit
    skip_color_bars = frame_config.brng_skip_color_bars == 'yes'
    max_refinements = frame_config.max_border_retries
    
    # Run enhanced analysis
    analyzer = EnhancedFrameAnalysis(video_path, output_dir)
    
    # Load existing border data if provided
    if border_data_path and Path(border_data_path).exists():
        with open(border_data_path, 'r') as f:
            border_data = json.load(f)
            # Would need to convert this to BorderDetectionResult
            # For now, proceed with fresh analysis
    
    results = analyzer.analyze(
        method=method,
        duration_limit=duration_limit,
        skip_color_bars=skip_color_bars,
        max_refinement_iterations=max_refinements
    )
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_frame_analysis.py <video_file> [output_dir]")
        sys.exit(1)
    
    video_file = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = analyze_frame_quality(video_file, output_dir=output_directory)
    print(f"\nAnalysis complete for {video_file}")