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
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import xml.etree.ElementTree as ET
from scipy import ndimage, signal
import logging

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
                logger.debug(f"    Period {period_num}: {frames_in_period:,} frames analyzed, "
                        f"{frames_with_violations:,} with violations ({violation_pct:.1f}%)")
                if frames_with_violations > 0:
                    logger.debug(f"    Period {period_num} max BRNG: {max_brng_value:.4f}%")
            else:
                logger.debug(f"    Period {period_num}: No frames found in time range {start_time:.1f}s - {end_time:.1f}s")
            
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
        black_frames_skipped = 0  # NEW COUNTER
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
                    
                    # Extract frame data - this now includes black frame detection
                    frame_data_before = frames_with_violations
                    frame_data = self._extract_frame_violations(elem, frame_num=None)
                    
                    # Check if this might have been a black frame
                    # (we can detect this by checking if no violation was returned despite BRNG being present)
                    brng_tag = elem.find('.//tag[@key="lavfi.signalstats.BRNG"]')
                    if brng_tag is not None and frame_data is None:
                        # Check if it was filtered due to being black
                        ymax_tag = elem.find('.//tag[@key="lavfi.signalstats.YMAX"]')
                        if ymax_tag:
                            ymax = float(ymax_tag.get('value', '1000'))
                            if ymax < 300.0:
                                black_frames_skipped += 1
                    
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
            logger.debug(f"  Checked {total_frames_checked:,} frames from QCTools report")
            if frames_skipped > 0:
                logger.debug(f"  Skipped {frames_skipped:,} color bar frames (first {color_bars_end_time:.1f}s)")
            if black_frames_skipped > 0:
                logger.debug(f"  Skipped {black_frames_skipped:,} all-black frames")
            
            if frames_after_color_bars > 0:
                violation_pct = (frames_with_violations / frames_after_color_bars * 100) if frames_after_color_bars > 0 else 0
                logger.debug(f"  Analyzed {frames_after_color_bars:,} content frames")
                logger.debug(f"  Found {frames_with_violations:,} frames with BRNG violations ({violation_pct:.1f}% of content)")
                if frames_with_violations > 0:
                    logger.debug(f"  Max BRNG value: {max_brng_value:.4f}%\n")
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
                frame_num_str = elem.get('n') or elem.get('pkt_pts')
                if frame_num_str:
                    frame_num = int(frame_num_str)
                else:
                    return None
            
            # Get timestamp
            timestamp_str = elem.get('pkt_pts_time')
            if timestamp_str:
                timestamp = float(timestamp_str)
            else:
                timestamp = frame_num / self.fps if frame_num else 0
            
            # Check for all-black frame BEFORE checking BRNG violations
            # Extract luma values to detect black frames
            ymax_tag = elem.find('.//tag[@key="lavfi.signalstats.YMAX"]')
            yhigh_tag = elem.find('.//tag[@key="lavfi.signalstats.YHIGH"]')
            ylow_tag = elem.find('.//tag[@key="lavfi.signalstats.YLOW"]')
            ymin_tag = elem.find('.//tag[@key="lavfi.signalstats.YMIN"]')
            
            # Check if this is an all-black frame
            if ymax_tag is not None and yhigh_tag is not None and ylow_tag is not None and ymin_tag is not None:
                ymax = float(ymax_tag.get('value', '1000'))
                yhigh = float(yhigh_tag.get('value', '1000'))
                ylow = float(ylow_tag.get('value', '1000'))
                ymin = float(ymin_tag.get('value', '1000'))
                
                # Skip all-black frames based on luma thresholds
                if ymax < 300.0 and yhigh < 115.0 and ylow < 97.0 and ymin < 6.5:
                    # This is a black frame, skip it
                    return None
            
            # Now check for BRNG violations (only for non-black frames)
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
        logger.debug(f"  Detected borders - L:{borders['left']}px R:{borders['right']}px T:{borders['top']}px B:{borders['bottom']}px")
        logger.debug(f"  Active picture area: {active_width}x{active_height} (from {self.width}x{self.height})")
        logger.debug(f"  Using {len(quality_frames)} quality frames for detection\n")
        
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
    
    def find_good_representative_frame(self, target_time=150, search_window=120):
        """
        Find a good representative frame using enhanced quality assessment
        
        Args:
            target_time: Target time in seconds for frame selection
            search_window: Window in seconds to search around target time
            
        Returns:
            Frame (numpy array) or None if no suitable frame found
        """
        # Calculate search range
        target_frame = int(target_time * self.fps)
        window_frames = int(search_window * self.fps)
        
        start_frame = max(0, target_frame - window_frames // 2)
        end_frame = min(self.total_frames - 1, target_frame + window_frames // 2)
        
        if end_frame >= self.total_frames:
            mid_point = self.total_frames // 2
            start_frame = max(0, mid_point - window_frames // 2)
            end_frame = min(self.total_frames - 1, mid_point + window_frames // 2)
        
        logger.debug(f"Searching for good representative frame between {start_frame/self.fps:.1f}s and {end_frame/self.fps:.1f}s...")
        
        # Open video for frame selection
        cap = cv2.VideoCapture(self.video_path)
        
        # Check frames every 1 second in the search window
        check_interval = max(1, int(self.fps))
        frame_indices = list(range(start_frame, end_frame, check_interval))
        
        # Use existing quality assessment to find the best frame
        quality_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Use existing _assess_frame_quality method
            quality = self._assess_frame_quality(frame)
            if quality['is_suitable']:
                quality_frames.append((idx, frame.copy(), quality['overall_quality']))
        
        cap.release()
        
        if quality_frames:
            # Sort by quality score and return the best frame
            quality_frames.sort(key=lambda x: x[2], reverse=True)
            best_frame_idx, best_frame, best_score = quality_frames[0]
            logger.info(f"✓ Selected high-quality frame at {best_frame_idx/self.fps:.1f}s (quality score: {best_score:.3f})\n")
            return best_frame
        else:
            # Final fallback - use target frame
            logger.warning(f"⚠️ No suitable frame found, using target frame as fallback\n")
            cap = cv2.VideoCapture(self.video_path)
            fallback_frame = target_frame if target_frame < self.total_frames else self.total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_frame)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        
    def generate_border_visualization(self, output_path, active_area=None, head_switching_results=None, target_time=150, search_window=120, detection_method='sophisticated'):
        """
        Generate visual showing detected borders and active area
        
        Creates a side-by-side comparison with:
        - Left: Full frame with border regions highlighted in red
        - Right: Active picture area only
        
        Args:
            output_path: Path to save the visualization image
            active_area: Tuple (x, y, w, h) of active picture area
            head_switching_results: Results from head switching analysis
            target_time: Target time for frame selection (seconds)
            search_window: Window to search for good frame (seconds)
            detection_method: Detection method used ('simple' or 'sophisticated')
            
        Returns:
            True if successful, False otherwise
        """
        # Import matplotlib (should be at top of file)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Find a good representative frame
        frame = self.find_good_representative_frame(target_time, search_window)
        
        if frame is None:
            logger.warning("Could not find suitable frame for border visualization")
            return False
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Full frame with regions marked
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax1.imshow(frame_rgb)
        
        # Add title indicating detection method
        if detection_method == 'simple':
            ax1.set_title('Full Frame with Border Detection\n(Simple Border Detection Used)', fontsize=10, color='darkred')
        else:
            ax1.set_title('Full Frame with Border Detection')
        ax1.axis('off')
        
        if active_area:
            x, y, w, h = active_area
            
            # Mark border regions differently based on detection method
            border_added = False
            
            if detection_method == 'simple':
                # Use dotted red lines for simple border detection
                line_style = '--'  # Dashed line style
                line_width = 2.5
                line_color = 'red'
                
                # Draw outline around active area with dotted red lines
                # Left border
                if x > 10:
                    ax1.plot([x, x], [0, self.height], linestyle=line_style, 
                            linewidth=line_width, color=line_color, label='Border Edge (Simple Detection)')
                    border_added = True
                
                # Right border
                if x + w < self.width - 10:
                    ax1.plot([x + w, x + w], [0, self.height], linestyle=line_style, 
                            linewidth=line_width, color=line_color)
                    if not border_added:
                        ax1.plot([x + w, x + w], [0, self.height], linestyle=line_style, 
                                linewidth=line_width, color=line_color, label='Border Edge (Simple Detection)')
                        border_added = True
                
                # Top border
                if y > 10:
                    ax1.plot([0, self.width], [y, y], linestyle=line_style, 
                            linewidth=line_width, color=line_color)
                    if not border_added:
                        ax1.plot([0, self.width], [y, y], linestyle=line_style, 
                                linewidth=line_width, color=line_color, label='Border Edge (Simple Detection)')
                        border_added = True
                
                # Bottom border
                if y + h < self.height - 10:
                    ax1.plot([0, self.width], [y + h, y + h], linestyle=line_style, 
                            linewidth=line_width, color=line_color)
                    if not border_added:
                        ax1.plot([0, self.width], [y + h, y + h], linestyle=line_style, 
                                linewidth=line_width, color=line_color, label='Border Edge (Simple Detection)')
                        border_added = True
            else:
                # Use shaded rectangles for sophisticated border detection
                if x > 10:  # Left border
                    left_rect = patches.Rectangle((0, 0), x, self.height, linewidth=2,
                                                edgecolor='red', facecolor='red', alpha=0.3,
                                                label='Border Regions')
                    ax1.add_patch(left_rect)
                    border_added = True
                
                if x + w < self.width - 10:  # Right border
                    right_rect = patches.Rectangle((x + w, 0), self.width - (x + w), self.height, 
                                                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
                    if not border_added:
                        right_rect.set_label('Border Regions')
                        border_added = True
                    ax1.add_patch(right_rect)
                    
                if y > 10:  # Top border
                    top_rect = patches.Rectangle((0, 0), self.width, y, linewidth=2,
                                            edgecolor='red', facecolor='red', alpha=0.3)
                    if not border_added:
                        top_rect.set_label('Border Regions')
                        border_added = True
                    ax1.add_patch(top_rect)
                    
                if y + h < self.height - 10:  # Bottom border
                    bottom_rect = patches.Rectangle((0, y + h), self.width, self.height - (y + h), 
                                                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
                    if not border_added:
                        bottom_rect.set_label('Border Regions')
                    ax1.add_patch(bottom_rect)
            
            # Highlight head switching artifacts if detected
            if head_switching_results and head_switching_results.get('detected'):
                # Draw orange line at bottom to indicate head switching artifacts
                # The current _detect_head_switching doesn't return detailed region info,
                # so we'll draw a line across the bottom of the frame
                line_y = self.height - 1
                ax1.plot([0, self.width], [line_y, line_y], 
                        color='orange', linewidth=3, alpha=0.9, 
                        label='Head Switching Artifacts')
            
            if border_added or (head_switching_results and head_switching_results.get('detected')):
                ax1.legend()
            
            # Active area only
            active_frame = frame_rgb[y:y+h, x:x+w]
            ax2.imshow(active_frame)
            ax2.set_title('Active Picture Area Only')
            ax2.axis('off')
            
            # Add text annotations
            border_info = f'Border sizes: L={x}px, R={self.width-x-w}px, T={y}px, B={self.height-y-h}px'
            fig.text(0.5, 0.02, border_info, ha='center', fontsize=10)
            
            # Add head switching info
            if head_switching_results and head_switching_results.get('detected'):
                severity = head_switching_results.get('severity', 'unknown')
                percentage = head_switching_results.get('percentage', 0)
                hs_info = f"Head switching artifacts: {severity} ({percentage:.1f}% of frames)"
                fig.text(0.5, 0.95, hs_info, ha='center', fontsize=12, weight='bold', color='orange')
            else:
                fig.text(0.5, 0.95, 'No head switching artifacts detected', ha='center', fontsize=12, weight='bold', color='green')
        else:
            ax2.text(0.5, 0.5, 'No borders detected', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
            
            # Still show head switching info even without borders
            if head_switching_results and head_switching_results.get('detected'):
                # Draw orange line at bottom
                line_y = self.height - 1
                ax1.plot([0, self.width], [line_y, line_y], 
                        color='orange', linewidth=3, alpha=0.9, 
                        label='Head Switching Artifacts')
                ax1.legend()
                
                severity = head_switching_results.get('severity', 'unknown')
                percentage = head_switching_results.get('percentage', 0)
                hs_info = f"Head switching artifacts: {severity} ({percentage:.1f}% of frames)"
                fig.text(0.5, 0.95, hs_info, ha='center', fontsize=12, weight='bold', color='orange')
            else:
                fig.text(0.5, 0.95, 'No head switching artifacts detected', ha='center', fontsize=12, weight='bold', color='green')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # logger.info(f"Border detection visualization saved to: {output_path}")
        return True


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
            # logger.debug(f"  Using {len(analysis_periods)} analysis periods from signalstats")
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
            logger.debug(f"  Analyzed {len(violations)} frames with potential violations across all periods")
        else:
            # Original single-period analysis (similar handling)
            logger.debug(f"  Creating temporary comparison videos (duration: {duration_limit}s)")
            
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
            
            # Clean up previous thumbnails to ensure only current run's thumbnails are shown
            for old_thumb in thumb_dir.glob("*.jpg"):
                try:
                    old_thumb.unlink()
                    logger.debug(f"  Removed previous thumbnail: {old_thumb.name}")
                except Exception as e:
                    logger.warning(f"  Could not remove old thumbnail {old_thumb.name}: {e}")
            for old_thumb in thumb_dir.glob("*.png"):
                try:
                    old_thumb.unlink()
                    logger.debug(f"  Removed previous thumbnail: {old_thumb.name}")
                except Exception as e:
                    logger.warning(f"  Could not remove old thumbnail {old_thumb.name}: {e}")
            
            # Select violations with temporal spacing
            selected_violations = self._select_diverse_violations_for_thumbnails(
                violations, 
                max_thumbnails=5, 
                min_time_separation=5.0
            )
            
            logger.debug(f"  Creating diagnostic thumbnails for {len(selected_violations)} temporally diverse violations\n")
            thumbnails = self._create_diagnostic_thumbnails(selected_violations, temp_video_paths, thumb_dir)
            logger.info(f"Saved {len(thumbnails)} thumbnails to {thumb_dir}")
        
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
            logger.debug(f"  Targeting {len(qctools_violations)} frames identified by QCTools")
            
            sample_indices = []
            for v in qctools_violations[:500]:  # Increased limit
                # Check if this violation is within our period
                if period_start_time <= v.timestamp < period_end_time:
                    # Convert to frame position in the extracted video segment
                    relative_time = v.timestamp - period_start_time
                    frame_in_segment = int(relative_time * fps)
                    if 0 <= frame_in_segment < total_frames:
                        sample_indices.append(frame_in_segment)
            
            logger.debug(f"  Mapped {len(sample_indices)} violation frames to processed video positions")
            
            # If we didn't get enough samples from QCTools, add some distributed samples
            if len(sample_indices) < 50:
                logger.debug(f"  Adding distributed samples to reach minimum coverage")
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
        
        logger.debug(f"  Analyzing {len(sample_indices)} frame samples...")
        
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
        
        logger.debug(f"  Checked {frames_checked} frames, found {len(violations)} with violations above threshold")
        
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
        
        logger.debug(f"  Final selection: {len(selected_violations)} thumbnails spanning {selected_violations[-1].timestamp - selected_violations[0].timestamp:.1f} seconds")
        
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
        """Generate detailed report based on violation patterns"""
        if not violations:
            return {
                'overall_assessment': 'No BRNG violations detected',
                'summary_statistics': {
                    'total_violations': 0,
                    'average_violation_percentage': 0,
                    'max_violation_percentage': 0,
                    'edge_violation_percentage': 0,
                    'linear_pattern_percentage': 0
                }
            }
        
        # Calculate statistics
        avg_violation_pct = np.mean([v.violation_percentage for v in violations])
        max_violation_pct = max([v.violation_percentage for v in violations])
        
        return {
            'overall_assessment': self._get_overall_assessment(violations, aggregate_patterns),
            'summary_statistics': {
                'total_violations': len(violations),
                'average_violation_percentage': avg_violation_pct,
                'max_violation_percentage': max_violation_pct,
                'edge_violation_percentage': aggregate_patterns.get('edge_violation_percentage', 0),
                'linear_pattern_percentage': aggregate_patterns.get('linear_pattern_percentage', 0)
            }
        }

    def _get_overall_assessment(self, violations: List[FrameViolation],
                        aggregate_patterns: Dict) -> str:
        """Generate descriptive assessment text based on analysis findings"""
        if not violations:
            return "No BRNG violations detected in analyzed frames"
        
        # Build description based on what was found
        edge_pct = aggregate_patterns.get('edge_violation_percentage', 0)
        linear_pct = aggregate_patterns.get('linear_pattern_percentage', 0)
        avg_violation = np.mean([v.violation_percentage for v in violations])
        
        parts = []
        
        # Describe violation level
        if avg_violation >= 1.0:
            parts.append(f"Average BRNG: {avg_violation:.2f}%")
        elif avg_violation >= 0.1:
            parts.append(f"Average BRNG: {avg_violation:.2f}% (low-level)")
        else:
            parts.append(f"Average BRNG: {avg_violation:.3f}% (minimal)")
        
        # Describe spatial distribution
        if edge_pct > 70:
            parts.append(f"{edge_pct:.0f}% of violations at frame edges")
        elif edge_pct > 40:
            parts.append(f"{edge_pct:.0f}% edge violations, {100-edge_pct:.0f}% content area")
        elif edge_pct > 0:
            parts.append(f"Violations primarily in content area ({100-edge_pct:.0f}%)")
        
        # Note linear patterns if present
        if linear_pct > 20:
            parts.append(f"linear blanking patterns detected ({linear_pct:.0f}%)")
        
        return "; ".join(parts)
    
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
        """Create violations display by extracting magenta pixels from the highlighted frame"""
        
        # Extract magenta pixels directly from the highlighted frame
        # This ensures we show exactly what ffmpeg's signalstats detected
        magenta_mask = self._extract_magenta_pixels(frame_h)
        
        # Create the visualization with yellow highlighting
        violations_only = np.zeros_like(frame_o)
        violations_only[magenta_mask > 0] = [0, 255, 255]  # Yellow in BGR
        
        # Optionally add slight transparency overlay for better visibility
        if np.any(magenta_mask > 0):
            violation_overlay = np.zeros_like(frame_o)
            violation_overlay[magenta_mask > 0] = frame_o[magenta_mask > 0] * 0.3
            violations_only = cv2.addWeighted(violations_only, 0.8, violation_overlay, 0.2, 0)
        
        return violations_only
    
    def _extract_magenta_pixels(self, frame: np.ndarray) -> np.ndarray:
        """Extract magenta-colored pixels from a frame (as added by ffmpeg signalstats filter)"""
        
        # Split channels (BGR format)
        b, g, r = cv2.split(frame)
        
        # Magenta has high blue and red, low green
        # Use multiple detection methods for robustness
        
        # Method 1: Direct BGR thresholding
        # Magenta should have B>threshold, R>threshold, G<threshold
        magenta_threshold = 200  # Fairly bright magenta
        green_threshold = 100    # Green should be lower
        
        bgr_magenta = (
            (b > magenta_threshold) & 
            (r > magenta_threshold) & 
            (g < green_threshold)
        )
        
        # Method 2: Ratio-based detection (R and B similar, G much lower)
        # Avoid division by zero
        safe_g = np.maximum(g, 1).astype(np.float32)
        rb_avg = ((r.astype(np.float32) + b.astype(np.float32)) / 2)
        
        ratio_magenta = (rb_avg / safe_g > 1.8) & (b > 150) & (r > 150)
        
        # Method 3: HSV-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Magenta hue is around 140-160 in OpenCV's 0-180 range
        # Also check near 0/180 (red-magenta boundary)
        hsv_magenta = (
            (((h >= 140) & (h <= 160)) | (h < 5) | (h > 175)) &
            (s > 100) &  # Saturated
            (v > 150)    # Bright
        )
        
        # Combine methods - use any 2 out of 3
        method1 = bgr_magenta.astype(np.uint8)
        method2 = ratio_magenta.astype(np.uint8)
        method3 = hsv_magenta.astype(np.uint8)
        
        vote_sum = method1 + method2 + method3
        magenta_mask = (vote_sum >= 2).astype(np.uint8) * 255
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return magenta_mask

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
    
    def __init__(self, video_path: str, qctools_report: str = None):
        self.video_path = str(video_path)
        self.qctools_report = qctools_report if qctools_report else self._find_qctools_report(log_result=False)
        self._init_video_properties()
        self._brng_cache = None
        self._brng_cache_active_area = None
        
    def _init_video_properties(self):
        """Initialize video properties"""
        cap = cv2.VideoCapture(self.video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()
    
    def _find_qctools_report(self, log_result: bool = True) -> Optional[str]:
        """Find existing QCTools report"""
        video_path = Path(self.video_path)
        video_id = video_path.stem
        video_filename = video_path.name  # Full filename with extension
        
        search_patterns = [
            # Try with full filename first (e.g., JPC_AV_01709.mkv.qctools.xml.gz)
            video_path.parent / f"{video_filename}.qctools.xml.gz",
            video_path.parent / f"{video_id}_qc_metadata" / f"{video_filename}.qctools.xml.gz",
            video_path.parent / f"{video_id}_vrecord_metadata" / f"{video_filename}.qctools.xml.gz",
            # Then try without extension (e.g., JPC_AV_01709.qctools.xml.gz)
            video_path.parent / f"{video_id}.qctools.xml.gz",
            video_path.parent / f"{video_id}_qc_metadata" / f"{video_id}.qctools.xml.gz",
            video_path.parent / f"{video_id}_vrecord_metadata" / f"{video_id}.qctools.xml.gz",
        ]
        
        for pattern in search_patterns:
            if pattern.exists():
                if log_result:
                    logger.debug(f"Found QCTools report for Signalstats Analyzer: {pattern}\n")
                return str(pattern)
        
        return None
    
    def analyze_with_signalstats(self,
                        border_data: BorderDetectionResult = None,
                        content_start_time: float = 0,
                        color_bars_end_time: float = None,
                        analysis_duration: int = 60,
                        num_periods: int = 3,
                        qctools_periods: List[Tuple[float, int]] = None) -> SignalstatsResult:
        """
        Analyze using signalstats, comparing full frame (QCTools) vs active area (FFprobe).
        
        Args:
            qctools_periods: Pre-computed periods from QCTools violation distribution analysis
        """
        # Determine analysis periods - now with QCTools priority
        analysis_periods = self._find_analysis_periods(
            content_start_time, 
            color_bars_end_time,
            analysis_duration,
            num_periods,
            border_data.quality_frame_hints if border_data else None,
            qctools_periods=qctools_periods  # NEW PARAMETER
        )
        
        # Log analysis configuration
        logger.info(f"  Running {len(analysis_periods)} analysis periods:")
        for i, (start_time, duration) in enumerate(analysis_periods):
            end_time = start_time + duration
            start_tc = self._seconds_to_timecode(start_time)
            end_tc = self._seconds_to_timecode(end_time)
            logger.debug(f"    Period {i+1}: {start_tc} - {end_tc} ({duration}s)")
        
        # Log active area vs full frame comparison
        active_area = border_data.active_area if border_data else None
        if active_area:
            x, y, w, h = active_area
            full_w, full_h = self.width, self.height
            crop_pct = (w * h) / (full_w * full_h) * 100
            logger.debug(f"  Comparison mode: Full frame ({full_w}x{full_h}) vs Active area ({w}x{h} at {x},{y})")
            logger.info(f"  Active area is {crop_pct:.1f}% of full frame\n")
        else:
            logger.debug(f"  Comparison mode: Full frame analysis only (no border detection)")
        
        # Analyze periods
        all_results = []
        used_qctools = False
        comparison_results = []
        
        for i, (start_time, duration) in enumerate(analysis_periods):
            logger.debug(f"  Analyzing period {i+1} ({self._seconds_to_timecode(start_time)} - {self._seconds_to_timecode(start_time + duration)}):")
            
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
                    logger.debug(f"    QCTools (full frame): {period_comparison['qctools_full_frame']['violations_pct']:.1f}% violations, "
                            f"max BRNG: {period_comparison['qctools_full_frame']['max_brng']:.4f}%")
            
            # Get FFprobe data (active area only)
            if active_area:
                logger.debug(f"    Running FFprobe on active area only...")
                ffprobe_result = self._analyze_with_ffprobe_period(
                    active_area, start_time, duration, i+1
                )
                if ffprobe_result:
                    period_comparison['ffprobe_active_area'] = {
                        'violations_pct': (ffprobe_result['frames_with_violations'] / 
                                        ffprobe_result['frames_analyzed'] * 100) if ffprobe_result['frames_analyzed'] > 0 else 0,
                        'max_brng': max(ffprobe_result['brng_values']) * 100 if ffprobe_result['brng_values'] else 0
                    }
                    logger.debug(f"    FFprobe (active area): {period_comparison['ffprobe_active_area']['violations_pct']:.1f}% violations, "
                            f"max BRNG: {period_comparison['ffprobe_active_area']['max_brng']:.4f}%")
                    
                    # Compare results
                    if qctools_result and period_comparison.get('qctools_full_frame'):
                        full_violations = period_comparison['qctools_full_frame']['violations_pct']
                        active_violations = period_comparison['ffprobe_active_area']['violations_pct']
                        
                        if full_violations > active_violations + 5:
                            logger.info(f"    → Border violations detected: Full frame has {full_violations - active_violations:.1f}% more violations\n")
                            period_comparison['diagnosis'] = 'border_violations'
                        elif active_violations > 10:
                            logger.info(f"    → Content violations: Active area itself has {active_violations:.1f}% violations\n")
                            period_comparison['diagnosis'] = 'content_violations'
                        else:
                            logger.info(f"    → Minimal violations in both full frame and active area\n")
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
        logger.info(f"  Diagnosis: {diagnosis}\n")
        
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
                      quality_hints: List[Tuple[float, float]] = None,
                      qctools_periods: List[Tuple[float, int]] = None) -> List[Tuple[float, int]]:
        """Find good analysis periods, preferring QCTools violation clusters when available"""
        
        # Start after color bars with a safety margin
        effective_start = max(content_start, color_bars_end or 0) + 10
        
        logger.debug(f"  Content starts at {effective_start:.1f}s (after color bars at {color_bars_end:.1f}s)")
        
        # PRIORITY 1: Use QCTools-based periods if available (these target actual violations)
        if qctools_periods:
            valid_periods = []
            for start_time, period_duration in qctools_periods:
                # Ensure period is after color bars
                adjusted_start = max(effective_start, start_time)
                if adjusted_start + period_duration <= self.duration - 10:
                    valid_periods.append((adjusted_start, period_duration))
            
            if len(valid_periods) >= num_periods:
                logger.info(f"  Using {num_periods} QCTools-based analysis periods (targeting violation clusters)")
                return valid_periods[:num_periods]
            elif valid_periods:
                # Use available QCTools periods and supplement with distributed periods
                logger.info(f"  Using {len(valid_periods)} QCTools-based periods, adding {num_periods - len(valid_periods)} distributed periods")
                
                # Calculate distributed periods for remaining slots
                available_duration = self.duration - effective_start - 30
                needed = num_periods - len(valid_periods)
                spacing = available_duration / (needed + 1)
                
                for i in range(needed):
                    candidate_start = effective_start + spacing * (i + 1)
                    # Avoid overlap with existing QCTools periods
                    overlaps = False
                    for existing_start, existing_dur in valid_periods:
                        if not (candidate_start + duration < existing_start or candidate_start > existing_start + existing_dur):
                            overlaps = True
                            break
                    if not overlaps:
                        valid_periods.append((candidate_start, duration))
                
                return valid_periods[:num_periods]
        
        # PRIORITY 2: Use quality hints if available (from border detection)
        if quality_hints:
            valid_hints = [(t, q) for t, q in quality_hints if t >= effective_start]
            if len(valid_hints) >= num_periods:
                periods = []
                for time_hint, _ in valid_hints[:num_periods]:
                    period_start = max(effective_start, time_hint - duration/2)
                    if period_start + duration <= self.duration - 30:
                        periods.append((period_start, duration))
                if len(periods) >= num_periods:
                    logger.info(f"  Using {len(periods)} quality hint-based analysis periods")
                    return periods
        
        # PRIORITY 3: Fall back to even distribution
        logger.info(f"  Using evenly distributed analysis periods (no violation clusters found)")
        available_duration = self.duration - effective_start - 30
        if available_duration < duration * num_periods:
            if available_duration >= duration:
                periods = []
                actual_periods = min(num_periods, int(available_duration / duration))
                spacing = (available_duration - duration) / max(1, actual_periods - 1)
                for i in range(actual_periods):
                    start = effective_start + i * spacing
                    periods.append((start, duration))
                return periods
            else:
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
        """Analyze using FFprobe signalstats for specific period - with fast seeking"""
        
        crop_filter = ""
        if active_area:
            x, y, w, h = active_area
            crop_filter = f"crop={w}:{h}:{x}:{y},"
        
        # Use movie filter's seek_point parameter for fast seeking
        # trim=duration limits output to N seconds after seek point
        filter_chain = f"movie={shlex.quote(self.video_path)}:seek_point={start_time}"
        filter_chain += f",{crop_filter}signalstats=stat=brng"
        filter_chain += f",trim=duration={duration}"
        
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
            
            logger.debug(f"    Period {period_num} FFprobe results: {len(brng_values):,} frames, "
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
    
    def __init__(self, video_path: str, output_dir: str = None, signals=None):
        self.video_path = Path(video_path)
        self.video_id = self.video_path.stem
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent
        self.output_dir.mkdir(exist_ok=True)
        self.signals = signals  # Store signals for emitting step completion

        # Store config manager as an instance attribute
        self.config_mgr = ConfigManager()
        self.config_mgr.refresh_configs()
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        
        # Find QCTools report once (with logging)
        self.qctools_report = self._find_qctools_report()
        if not self.qctools_report:
            logger.warning(f"No QCTools report found for {self.video_path.name}")
        
        # Initialize components (pass qctools_report to avoid duplicate search)
        self.border_detector = SophisticatedBorderDetector(video_path)
        self.brng_analyzer = None  # Will be initialized with border data
        self.signalstats_analyzer = IntegratedSignalstatsAnalyzer(video_path, qctools_report=self.qctools_report)
        
        # Initialize QCTools parser if report exists
        self.qctools_parser = None
        if self.qctools_report:
            self.qctools_parser = QCToolsParser(self.qctools_report)
    
    def _find_qctools_report(self) -> Optional[str]:
        """Find QCTools report for the video"""
        # Get the full filename with extension
        video_filename = self.video_path.name
        
        search_paths = [
            # Try with full filename first (e.g., JPC_AV_01709.mkv.qctools.xml.gz)
            self.video_path.parent / f"{video_filename}.qctools.xml.gz",
            self.video_path.parent / f"{self.video_id}_qc_metadata" / f"{video_filename}.qctools.xml.gz",
            self.video_path.parent / f"{self.video_id}_vrecord_metadata" / f"{video_filename}.qctools.xml.gz",
            # Then try without extension (e.g., JPC_AV_01709.qctools.xml.gz)
            self.video_path.parent / f"{self.video_id}.qctools.xml.gz",
            self.video_path.parent / f"{self.video_id}_qc_metadata" / f"{self.video_id}.qctools.xml.gz",
            self.video_path.parent / f"{self.video_id}_vrecord_metadata" / f"{self.video_id}.qctools.xml.gz",
        ]
        
        for path in search_paths:
            if path.exists():
                logger.debug(f"Found QCTools report for Frame Analysis: {path}\n")
                return str(path)
        
        return None
    
    def _is_step_enabled(self, flag_value) -> bool:
        """
        Check if a step is enabled, handling both boolean and string types.
        
        Args:
            flag_value: Can be bool (True/False) or str ("yes"/"no")
            
        Returns:
            bool: True if the step should run, False otherwise
        """
        if isinstance(flag_value, bool):
            return flag_value
        elif isinstance(flag_value, str):
            return flag_value.lower() in ('yes', 'true', '1')
        else:
            # Default to True for unknown types to maintain backward compatibility
            return True
        
    def _get_video_duration(self) -> Optional[float]:
        """Get video duration in seconds using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                str(self.video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
        return None
    
    def analyze(self, 
        method: str = 'sophisticated',
        duration_limit: int = 300,
        skip_color_bars: bool = True,
        max_refinement_iterations: int = 3,
        color_bars_end_time: float = None,
        signals=None) -> Dict:
        """
        Run complete enhanced frame analysis with optional iterative refinement.
        
        Args:
            method: 'sophisticated' or 'simple' border detection
            duration_limit: Maximum duration to analyze (seconds)
            skip_color_bars: Whether to skip color bars at start
            max_refinement_iterations: Maximum border refinement iterations
            color_bars_end_time: End time of color bars if detected
            signals: Optional signals object for emitting step completion events
        
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
        
        # Check which steps are enabled (handle both bool and str types)
        border_detection_enabled = self._is_step_enabled(frame_config.enable_border_detection)
        brng_analysis_enabled = self._is_step_enabled(frame_config.enable_brng_analysis)
        signalstats_enabled = self._is_step_enabled(frame_config.enable_signalstats)
        
        # Log which steps will run
        logger.info(f"Frame analysis configuration:")
        logger.debug(f"  Border detection: {'enabled' if border_detection_enabled else 'disabled'}")
        logger.debug(f"  BRNG analysis: {'enabled' if brng_analysis_enabled else 'disabled'}")
        logger.debug(f"  Signalstats: {'enabled' if signalstats_enabled else 'disabled'}\n")
        
        # Track what was actually run
        results['steps_enabled'] = {
            'border_detection': border_detection_enabled,
            'brng_analysis': brng_analysis_enabled,
            'signalstats': signalstats_enabled
        }
        
        # Step 1: Detect color bars (always run if skip_color_bars is True, as it's a prerequisite)
        if skip_color_bars:
            if color_bars_end_time is None:
                color_bars_end_time = self._detect_color_bars_duration()
                # Only log if we detected color bars via fallback (not already passed in)
                if color_bars_end_time > 0:
                    logger.info(f"Color bars detected, ending at {color_bars_end_time:.1f}s\n")
            if color_bars_end_time > 0:
                results['color_bars_end_time'] = color_bars_end_time

        # Step 2: Parse QCTools for initial violations (needed for BRNG analysis or border detection)
        violations = []
        if self.qctools_report :
            logger.info("Parsing QCTools report for violations...")
            parser = QCToolsParser(self.qctools_report )
            violations = parser.parse_for_violations_streaming(
                max_frames=100,
                skip_color_bars=skip_color_bars,
                color_bars_end_time=color_bars_end_time
            )
            frames_with_qctools_violations = len(violations)
            
            if frames_with_qctools_violations == 0:
                results['qctools_violations_found'] = "No BRNG violations detected in content"
            else:
                results['qctools_violations_found'] = frames_with_qctools_violations
        elif not self.qctools_parser:
            logger.info("No QCTools report found")
        else:
            logger.info("Skipping QCTools parsing (neither BRNG analysis nor border detection enabled)")

        # Analyze QCTools violation distribution to find optimal analysis periods
        qctools_suggested_periods = []
        if violations:
            qctools_suggested_periods = self._analyze_qctools_violation_distribution(violations)
            logger.info(f"Identified {len(qctools_suggested_periods)} periods with highest violation density\n")
        
        # Step 3: Border detection (conditional)
        border_results = None
        if border_detection_enabled:
            logger.info(f"Detecting borders using {method} method...")
            border_results = self.border_detector.detect_borders_with_quality_assessment(
                violations=violations,
                method=method
            )
            results['initial_borders'] = asdict(border_results)

            # Create border detection visualization
            logger.info("Creating border detection visualization...\n")
            viz_filename = f"{self.video_id}_border_detection.jpg"
            viz_output_path = self.output_dir / viz_filename
            try:
                success = self.border_detector.generate_border_visualization(
                    output_path=str(viz_output_path),
                    active_area=border_results.active_area,
                    head_switching_results=border_results.head_switching_artifacts,
                    target_time=150,  # Default to 2.5 minutes in
                    search_window=120,  # Search within 2 minutes
                    detection_method=border_results.detection_method
                )
                if success:
                    results['border_visualization'] = str(viz_output_path)
                    logger.info(f"✓ Border visualization saved to: {viz_output_path}\n")
                else:
                    logger.warning("Failed to generate border visualization")
            except Exception as e:
                logger.warning(f"Error creating border visualization: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # Emit border detection completion signal
            if signals and frame_config.enable_border_detection:
                signals.step_completed.emit("Frame Analysis - Border Detection")
        else:
            logger.warning("Skipping border detection (disabled in config)")
            # Create a default border result for downstream steps if needed
            if brng_analysis_enabled or signalstats_enabled:
                # Use full frame as "active area" when border detection is disabled
                logger.debug("Using full frame dimensions as active area")
                border_results = BorderDetectionResult(
                    active_area=(0, 0, self.border_detector.width, self.border_detector.height),
                    border_regions={},
                    detection_method='disabled',
                    quality_frame_hints=[],
                    head_switching_artifacts=None,
                    requires_refinement=False,
                    expansion_recommendations=None
                )
                results['initial_borders'] = asdict(border_results)
        
        # Step 4: Signalstats analysis (conditional)
        signalstats_results = None
        analysis_periods = []
        if signalstats_enabled:
            logger.info("Running signalstats analysis on active picture area to identify key analysis periods...")
            signalstats_results = self.signalstats_analyzer.analyze_with_signalstats(
                border_data=border_results,
                content_start_time=color_bars_end_time + 10 if color_bars_end_time else 10,
                color_bars_end_time=color_bars_end_time,
                analysis_duration=frame_config.signalstats_duration,
                num_periods=frame_config.signalstats_periods,
                qctools_periods=qctools_suggested_periods
            )
            results['signalstats'] = asdict(signalstats_results)

            # Extract the analysis periods from signalstats results
            analysis_periods = signalstats_results.analysis_periods
            logger.info(f"Identified {len(analysis_periods)} key periods for BRNG analysis")

            # Compare with QCTools violation distribution if we have violations
            if violations and qctools_suggested_periods:
                logger.info("\n  === Comparing Period Selection Methods ===")
                
                # Log the signalstats periods for comparison
                logger.debug(f"\n  Signalstats-selected analysis periods:")
                for i, (start, duration) in enumerate(analysis_periods):
                    logger.debug(f"    Period {i+1}: {start:.1f}s - {start+duration:.1f}s")
                
                # Check for overlaps
                logger.debug(f"\n  Checking for overlaps between methods:")
                for i, (ss_start, ss_duration) in enumerate(analysis_periods):
                    ss_end = ss_start + ss_duration
                    overlaps = []
                    
                    for j, (qct_start, qct_duration) in enumerate(qctools_suggested_periods):
                        qct_end = qct_start + qct_duration
                        # Check if periods overlap
                        if not (ss_end < qct_start or ss_start > qct_end):
                            overlap_start = max(ss_start, qct_start)
                            overlap_end = min(ss_end, qct_end)
                            overlap_duration = overlap_end - overlap_start
                            overlaps.append((j+1, overlap_duration))
                    
                    if overlaps:
                        overlap_str = ", ".join([f"QCT Period {j} ({dur:.1f}s)" for j, dur in overlaps])
                        logger.debug(f"    Signalstats Period {i+1} overlaps with: {overlap_str}")
                    else:
                        logger.debug(f"    Signalstats Period {i+1}: NO OVERLAP with QCTools violations")
                
                # Count violations in each signalstats period
                logger.info(f"\n  Actual QCTools violations in signalstats periods:")
                for i, (start, duration) in enumerate(analysis_periods):
                    end = start + duration
                    period_violations = [v for v in violations if start <= v.timestamp < end]
                    logger.debug(f"    Period {i+1} ({start:.1f}s - {end:.1f}s): {len(period_violations)} violations")
                    if period_violations and len(period_violations) <= 5:
                        # Show the actual timestamps if there are only a few
                        for v in period_violations:
                            logger.debug(f"      - Violation at {v.timestamp:.1f}s")
            
            # Emit signalstats completion signal
            if signals and frame_config.enable_signalstats:
                signals.step_completed.emit("Frame Analysis - Signalstats")
        else:
            logger.warning("Skipping signalstats analysis (disabled in config)\n")
        
        # Step 5: BRNG analysis (conditional)
        brng_results = None
        if brng_analysis_enabled:
            # If signalstats wasn't run, create default analysis periods
            if not analysis_periods:
                logger.info(f"Creating default analysis periods for BRNG highlights (signalstats was disabled)\n")
                # Create evenly distributed periods across the video
                video_duration = self._get_video_duration()
                if video_duration:
                    content_start = color_bars_end_time + 10  # Start 10s after color bars
                    content_duration = video_duration - content_start - 10  # Leave 10s at end
                    if content_duration > 0:
                        period_duration = frame_config.signalstats_duration
                        num_periods = min(frame_config.signalstats_periods, 3)
                        spacing = content_duration / (num_periods + 1)
                        for i in range(num_periods):
                            start_time = content_start + spacing * (i + 1)
                            analysis_periods.append((start_time, period_duration))
                        logger.debug(f"Created {len(analysis_periods)} default BRNG analysis periods\n")
            
            logger.info("\nAnalyzing BRNG violations in identified periods...")
            self.brng_analyzer = DifferentialBRNGAnalyzer(self.video_path, border_results)
            
            brng_results = self.brng_analyzer.analyze_with_differential_detection(
                output_dir=self.output_dir, 
                duration_limit=duration_limit,
                skip_start_seconds=color_bars_end_time,
                qctools_violations=violations,
                analysis_periods=analysis_periods 
            )
            results['brng_analysis'] = asdict(brng_results) if brng_results else None
            
            # Emit BRNG analysis completion signal
            if signals and frame_config.enable_brng_analysis:
                signals.step_completed.emit("Frame Analysis - BRNG Analysis")
            # Log BRNG analysis summary
            if brng_results:
                self._log_brng_analysis_summary(brng_results, analysis_periods)
        else:
            logger.warning("Skipping BRNG analysis (disabled in config)\n")
            # Log correlation between signalstats and BRNG analysis
            if signalstats_results and brng_results:
                self._log_analysis_correlation(signalstats_results, brng_results)
        
        # Step 6: Iterative border refinement (only if both border detection AND BRNG analysis are enabled)
        refinement_iterations = 0
        refinement_history = []  # Track all iterations for comparison

        if (border_detection_enabled and brng_analysis_enabled and 
            method == 'sophisticated' and brng_results and brng_results.requires_border_adjustment):
            logger.info("Border refinement needed - detected BRNG violations at frame edges")
            
            edge_pct = brng_results.aggregate_patterns.get('edge_violation_percentage', 0)
            continuous_pct = brng_results.aggregate_patterns.get('continuous_edge_percentage', 0)
            logger.info(f"  Edge violations: {edge_pct:.1f}% of analyzed frames")
            logger.info(f"  Continuous edge patterns: {continuous_pct:.1f}% of analyzed frames")
            
            # Check if auto_retry is enabled
            auto_retry_enabled = self._is_step_enabled(frame_config.auto_retry_borders)
            
            if not auto_retry_enabled:
                logger.info("Auto-retry borders is disabled, skipping refinement loop")
            else:
                # Store initial state for comparison
                initial_borders = border_results
                initial_brng = brng_results
                
                while (refinement_iterations < max_refinement_iterations and 
                    brng_results.requires_border_adjustment):
                    
                    refinement_iterations += 1
                    logger.info(f"Refinement iteration {refinement_iterations}/{max_refinement_iterations}:")
                    
                    # Store pre-refinement state
                    previous_area = border_results.active_area
                    previous_brng = brng_results
                    
                    # Refine borders
                    border_results = self.border_detector.refine_borders(border_results, brng_results)
                    new_area = border_results.active_area
                    
                    logger.info(f"  Active area: {previous_area[2]}x{previous_area[3]} → {new_area[2]}x{new_area[3]}")
                    
                    # CREATE VISUALIZATION FOR THIS ITERATION
                    logger.info("  Creating border visualization for refined borders...")
                    viz_filename = f"{self.video_id}_border_detection_refined_iter{refinement_iterations}.jpg"
                    viz_output_path = self.output_dir / viz_filename
                    
                    try:
                        success = self.border_detector.generate_border_visualization(
                            output_path=str(viz_output_path),
                            active_area=border_results.active_area,
                            head_switching_results=border_results.head_switching_artifacts,
                            target_time=150,
                            search_window=120,
                            detection_method=border_results.detection_method
                        )
                        
                        if success:
                            logger.info(f"  ✓ Refined border visualization saved: {viz_filename}")
                        else:
                            logger.warning(f"  ⚠ Failed to create visualization for iteration {refinement_iterations}")
                    except Exception as e:
                        logger.warning(f"  ⚠ Error creating visualization: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                    
                    # Re-run signalstats with new borders to get updated periods (only if signalstats is enabled)
                    if signalstats_enabled:
                        logger.info("  Re-running signalstats with refined borders...")
                        signalstats_results = self.signalstats_analyzer.analyze_with_signalstats(
                            border_data=border_results,
                            content_start_time=0,
                            color_bars_end_time=color_bars_end_time,
                            analysis_duration=frame_config.signalstats_duration,
                            num_periods=frame_config.signalstats_periods,
                            qctools_periods=qctools_suggested_periods
                        )
                        analysis_periods = signalstats_results.analysis_periods
                    
                    # Re-analyze BRNG with new borders and periods
                    logger.info("  Re-analyzing BRNG violations with refined borders...")
                    self.brng_analyzer = DifferentialBRNGAnalyzer(self.video_path, border_results)
                    
                    brng_results = self.brng_analyzer.analyze_with_differential_detection(
                        output_dir=self.output_dir,
                        duration_limit=duration_limit,
                        skip_start_seconds=color_bars_end_time,
                        qctools_violations=violations,
                        analysis_periods=analysis_periods
                    )
                    
                    # Track this iteration's results
                    iteration_data = {
                        'iteration': refinement_iterations,
                        'active_area': new_area,
                        'area_change': {
                            'width': new_area[2] - previous_area[2],
                            'height': new_area[3] - previous_area[3]
                        },
                        'violations_before': len(previous_brng.violations) if previous_brng.violations else 0,
                        'violations_after': len(brng_results.violations) if brng_results.violations else 0,
                        'edge_violation_pct': brng_results.aggregate_patterns.get('edge_violation_percentage', 0),
                        'visualization_path': str(viz_output_path) if success else None
                    }
                    refinement_history.append(iteration_data)
                    
                    # Log improvement metrics
                    violation_reduction = iteration_data['violations_before'] - iteration_data['violations_after']
                    if violation_reduction > 0:
                        logger.info(f"  Violations reduced: {iteration_data['violations_before']} → {iteration_data['violations_after']} (-{violation_reduction})")
                    else:
                        logger.info(f"  Violations: {iteration_data['violations_after']} (no reduction)")
                    
                    # Check for improvement
                    improved = self._is_meaningful_improvement(previous_brng, brng_results)
                    if not improved:
                        logger.info("  Stopping refinement - further adjustments unlikely to help")
                        break
                
                # After refinement loop completes
                results['refinement_iterations'] = refinement_iterations
                results['refinement_history'] = refinement_history
                results['final_borders'] = asdict(border_results)
                results['final_brng_analysis'] = asdict(brng_results) if brng_results else None
                if signalstats_enabled:
                    results['final_signalstats'] = asdict(signalstats_results)
                
                # CREATE COMPARISON VISUALIZATION (initial vs final)
                if refinement_iterations > 0:
                    logger.info("\nCreating before/after comparison visualization...")
                    comparison_path = self.output_dir / f"{self.video_id}_border_refinement_comparison.jpg"
                    
                    try:
                        self._create_refinement_comparison(
                            initial_borders=initial_borders,
                            final_borders=border_results,
                            initial_brng=initial_brng,
                            final_brng=brng_results,
                            output_path=comparison_path,
                            refinement_history=refinement_history
                        )
                        logger.info(f"✓ Refinement comparison saved: {comparison_path.name}")
                        results['refinement_comparison'] = str(comparison_path)
                    except Exception as e:
                        logger.warning(f"Could not create refinement comparison: {e}")
                
                # Summary of refinement
                logger.info(f"\n{'='*60}")
                logger.info(f"Border Refinement Summary:")
                logger.info(f"  Iterations performed: {refinement_iterations}")
                
                if refinement_iterations > 0:
                    initial_area = initial_borders.active_area
                    final_area = border_results.active_area
                    width_change = final_area[2] - initial_area[2]
                    height_change = final_area[3] - initial_area[3]
                    
                    logger.info(f"  Initial active area: {initial_area[2]}x{initial_area[3]}")
                    logger.info(f"  Final active area: {final_area[2]}x{final_area[3]}")
                    logger.info(f"  Size change: width {width_change:+d}px, height {height_change:+d}px")
                    
                    initial_violations = len(initial_brng.violations) if initial_brng.violations else 0
                    final_violations = len(brng_results.violations) if brng_results.violations else 0
                    violation_reduction = initial_violations - final_violations
                    
                    logger.info(f"  Violations: {initial_violations} → {final_violations} ({violation_reduction:+d})")
                    
                    # Calculate improvement percentage
                    if initial_violations > 0:
                        improvement_pct = (violation_reduction / initial_violations) * 100
                        logger.info(f"  Improvement: {improvement_pct:.1f}%")
                
                logger.info(f"{'='*60}\n")
        
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
    
    def _create_refinement_comparison(self, initial_borders, final_borders, 
                                 initial_brng, final_brng, output_path, 
                                 refinement_history):
        """
        Create a comparison visualization showing before and after refinement.
        
        This creates a side-by-side comparison showing:
        - Left: Initial border detection
        - Right: Final refined borders
        - Annotations showing the changes and improvements
        
        Args:
            initial_borders: BorderDetectionResult before refinement
            final_borders: BorderDetectionResult after refinement
            initial_brng: BRNGAnalysisResult before refinement
            final_brng: BRNGAnalysisResult after refinement
            output_path: Path to save the comparison image
            refinement_history: List of iteration data dictionaries
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Get a good frame for visualization
        frame = self.border_detector.find_good_representative_frame(
            target_time=150,
            search_window=120
        )
        
        if frame is None:
            logger.warning("Could not find frame for refinement comparison")
            return False
        
        # Create figure with 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # LEFT: Initial borders
        ax1.imshow(frame_rgb)
        ax1.set_title('Initial Border Detection', fontsize=14, weight='bold')
        ax1.axis('off')
        
        x1, y1, w1, h1 = initial_borders.active_area
        
        if initial_borders.active_area:
            # Draw initial borders in red
            if x1 > 10:
                left_rect = patches.Rectangle((0, 0), x1, self.border_detector.height,
                                            linewidth=2, edgecolor='red', 
                                            facecolor='red', alpha=0.3,
                                            label='Initial Borders')
                ax1.add_patch(left_rect)
            
            if x1 + w1 < self.border_detector.width - 10:
                right_rect = patches.Rectangle((x1 + w1, 0), 
                                            self.border_detector.width - (x1 + w1), 
                                            self.border_detector.height,
                                            linewidth=2, edgecolor='red', 
                                            facecolor='red', alpha=0.3)
                ax1.add_patch(right_rect)
            
            if y1 > 10:
                top_rect = patches.Rectangle((0, 0), self.border_detector.width, y1,
                                            linewidth=2, edgecolor='red', 
                                            facecolor='red', alpha=0.3)
                ax1.add_patch(top_rect)
            
            if y1 + h1 < self.border_detector.height - 10:
                bottom_rect = patches.Rectangle((0, y1 + h1), 
                                            self.border_detector.width, 
                                            self.border_detector.height - (y1 + h1),
                                            linewidth=2, edgecolor='red', 
                                            facecolor='red', alpha=0.3)
                ax1.add_patch(bottom_rect)
            
            # Draw active area outline in green
            active_rect = patches.Rectangle((x1, y1), w1, h1,
                                        linewidth=3, edgecolor='green', 
                                        facecolor='none',
                                        label='Active Area')
            ax1.add_patch(active_rect)
            ax1.legend(loc='upper right')
        
        # RIGHT: Final borders
        ax2.imshow(frame_rgb)
        ax2.set_title('After Refinement', fontsize=14, weight='bold')
        ax2.axis('off')
        
        if final_borders.active_area:
            x2, y2, w2, h2 = final_borders.active_area
            
            # Draw final borders in red
            if x2 > 10:
                left_rect = patches.Rectangle((0, 0), x2, self.border_detector.height,
                                            linewidth=2, edgecolor='red', 
                                            facecolor='red', alpha=0.3,
                                            label='Refined Borders')
                ax2.add_patch(left_rect)
            
            if x2 + w2 < self.border_detector.width - 10:
                right_rect = patches.Rectangle((x2 + w2, 0), 
                                            self.border_detector.width - (x2 + w2), 
                                            self.border_detector.height,
                                            linewidth=2, edgecolor='red', 
                                            facecolor='red', alpha=0.3)
                ax2.add_patch(right_rect)
            
            if y2 > 10:
                top_rect = patches.Rectangle((0, 0), self.border_detector.width, y2,
                                            linewidth=2, edgecolor='red', 
                                            facecolor='red', alpha=0.3)
                ax2.add_patch(top_rect)
            
            if y2 + h2 < self.border_detector.height - 10:
                bottom_rect = patches.Rectangle((0, y2 + h2), 
                                            self.border_detector.width, 
                                            self.border_detector.height - (y2 + h2),
                                            linewidth=2, edgecolor='red', 
                                            facecolor='red', alpha=0.3)
                ax2.add_patch(bottom_rect)
            
            # Draw active area outline in green
            active_rect = patches.Rectangle((x2, y2), w2, h2,
                                        linewidth=3, edgecolor='green', 
                                        facecolor='none',
                                        label='Active Area')
            ax2.add_patch(active_rect)
            
            # If borders changed significantly, draw arrows showing the change
            if abs(x2 - x1) > 5:  # Left border moved
                mid_y = self.border_detector.height // 2
                if x2 < x1:  # Border expanded left
                    ax2.annotate('', xy=(x2, mid_y), xytext=(x1, mid_y),
                            arrowprops=dict(arrowstyle='<-', color='cyan', lw=3))
                else:  # Border contracted right
                    ax2.annotate('', xy=(x2, mid_y), xytext=(x1, mid_y),
                            arrowprops=dict(arrowstyle='->', color='cyan', lw=3))
            
            ax2.legend(loc='upper right')
        
        # Add comprehensive comparison text
        comparison_text = self._format_refinement_comparison_text(
            initial_borders, final_borders, initial_brng, final_brng, refinement_history
        )
        
        # Add text box with comparison details
        fig.text(0.5, 0.02, comparison_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add main title
        iterations = len(refinement_history)
        title = f'Border Refinement: {iterations} Iteration{"s" if iterations != 1 else ""}'
        fig.text(0.5, 0.97, title, ha='center', fontsize=16, weight='bold')
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True


    def _format_refinement_comparison_text(self, initial_borders, final_borders, 
                                        initial_brng, final_brng, refinement_history):
        """
        Format the comparison text for the refinement visualization.
        
        Returns:
            Formatted string with comparison details
        """
        x1, y1, w1, h1 = initial_borders.active_area
        x2, y2, w2, h2 = final_borders.active_area
        
        # Calculate changes
        width_change = w2 - w1
        height_change = h2 - h1
        left_change = x2 - x1
        right_change = (self.border_detector.width - (x2 + w2)) - (self.border_detector.width - (x1 + w1))
        top_change = y2 - y1
        bottom_change = (self.border_detector.height - (y2 + h2)) - (self.border_detector.height - (y1 + h1))
        
        # BRNG violations
        initial_violations = len(initial_brng.violations) if initial_brng.violations else 0
        final_violations = len(final_brng.violations) if final_brng.violations else 0
        violation_reduction = initial_violations - final_violations
        
        # Build text
        lines = []
        lines.append(f"Active Area: {w1}x{h1} → {w2}x{h2} (Δ width: {width_change:+d}px, Δ height: {height_change:+d}px)")
        
        # Border changes
        border_changes = []
        if abs(left_change) > 2:
            direction = "expanded" if left_change < 0 else "contracted"
            border_changes.append(f"Left {direction} {abs(left_change)}px")
        if abs(right_change) > 2:
            direction = "expanded" if right_change > 0 else "contracted"
            border_changes.append(f"Right {direction} {abs(right_change)}px")
        if abs(top_change) > 2:
            direction = "expanded" if top_change < 0 else "contracted"
            border_changes.append(f"Top {direction} {abs(top_change)}px")
        if abs(bottom_change) > 2:
            direction = "expanded" if bottom_change > 0 else "contracted"
            border_changes.append(f"Bottom {direction} {abs(bottom_change)}px")
        
        if border_changes:
            lines.append(f"Border Changes: {', '.join(border_changes)}")
        else:
            lines.append("Border Changes: None")
        
        # Violation improvement
        lines.append(f"BRNG Violations: {initial_violations} → {final_violations} ({violation_reduction:+d})")
        
        if initial_violations > 0:
            improvement_pct = (violation_reduction / initial_violations) * 100
            lines.append(f"Improvement: {improvement_pct:.1f}%")
        
        # Edge violation percentages
        initial_edge_pct = initial_brng.aggregate_patterns.get('edge_violation_percentage', 0)
        final_edge_pct = final_brng.aggregate_patterns.get('edge_violation_percentage', 0)
        lines.append(f"Edge Violations: {initial_edge_pct:.1f}% → {final_edge_pct:.1f}%")
        
        return " | ".join(lines)
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate comprehensive human-readable summary with specific details"""
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
                    lines.append(f"  Frames with BRNG > 0: {violations}")
        
        # Border detection with frame dimensions
        if 'initial_borders' in results:
            borders = results.get('final_borders', results['initial_borders'])
            if borders['active_area']:
                x, y, w, h = borders['active_area']
                lines.append(f"\nBorder Detection:")
                lines.append(f"  Active area: {w}x{h} at ({x},{y})")
                lines.append(f"  Method: {borders['detection_method']}")
                # Calculate border widths
                frame_w, frame_h = 720, 486  # Standard NTSC
                left_border = x
                right_border = frame_w - (x + w)
                top_border = y
                bottom_border = frame_h - (y + h)
                lines.append(f"  Borders: L:{left_border}px R:{right_border}px T:{top_border}px B:{bottom_border}px")
        
        # BRNG analysis statistics
        if results.get('brng_analysis'):
            brng = results.get('final_brng_analysis', results['brng_analysis'])
            stats = brng.get('actionable_report', {}).get('summary_statistics', {})
            aggregate = brng.get('aggregate_patterns', {})
            
            lines.append(f"\nBRNG Analysis:")
            lines.append(f"  Frames analyzed: {stats.get('total_violations', 0)}")
            lines.append(f"  Average BRNG: {stats.get('average_violation_percentage', 0):.2f}%")
            lines.append(f"  Maximum BRNG: {stats.get('max_violation_percentage', 0):.2f}%")
            lines.append(f"  Edge violations: {stats.get('edge_violation_percentage', 0):.1f}% of analyzed frames")
            
            linear_pct = stats.get('linear_pattern_percentage', 0)
            if linear_pct > 0:
                lines.append(f"  Linear patterns: {linear_pct:.1f}% of analyzed frames")
            
            # Diagnostic breakdown
            if brng.get('violations'):
                diagnostic_counts = {}
                for v in brng['violations']:
                    if isinstance(v, dict) and v.get('diagnostics'):
                        for diag in v['diagnostics']:
                            if diag.startswith("Edge artifacts"):
                                diagnostic_counts["Edge artifacts"] = diagnostic_counts.get("Edge artifacts", 0) + 1
                            elif diag != "Border adjustment recommended":
                                diagnostic_counts[diag] = diagnostic_counts.get(diag, 0) + 1
                
                if diagnostic_counts:
                    lines.append(f"  Violation types:")
                    for diag, count in sorted(diagnostic_counts.items(), key=lambda x: x[1], reverse=True):
                        lines.append(f"    {diag}: {count} frames")
        
        # Signalstats comparison
        if results.get('signalstats'):
            stats = results['signalstats']
            lines.append(f"\nSignalstats (active area analysis):")
            lines.append(f"  Frames with violations: {stats['violation_percentage']:.1f}%")
            lines.append(f"  Max BRNG: {stats['max_brng']:.2f}%")
            lines.append(f"  Avg BRNG: {stats['avg_brng']:.2f}%")
            
            # Add analysis period info
            periods = stats.get('analysis_periods', [])
            if periods:
                period_strs = [f"{p[0]:.1f}s-{p[0]+p[1]:.1f}s" for p in periods]
                lines.append(f"  Analysis periods: {', '.join(period_strs)}")
        
        # Refinement info
        if 'refinement_iterations' in results and results['refinement_iterations'] > 0:
            lines.append(f"\nBorder refinement performed: {results['refinement_iterations']} iteration(s)")
            if 'initial_borders' in results and 'final_borders' in results:
                initial = results['initial_borders']['active_area']
                final = results['final_borders']['active_area']
                if initial and final:
                    w_change = final[2] - initial[2]
                    h_change = final[3] - initial[3]
                    lines.append(f"  Size change: width {w_change:+d}px, height {h_change:+d}px")
        
        return "\n".join(lines)
    
    def _log_brng_analysis_summary(self, brng_results: BRNGAnalysisResult, 
                                analysis_periods: List[Tuple[float, int]]) -> None:
        """Log a comprehensive summary of BRNG analysis results."""
        if not brng_results or not brng_results.violations:
            logger.info("\n  === BRNG Frame Analysis Summary ===")
            logger.info("  No BRNG violations detected in analyzed periods.\n")
            return
        
        violations = brng_results.violations
        aggregate = brng_results.aggregate_patterns
        report = brng_results.actionable_report
        stats = report.get('summary_statistics', {})
        
        # Calculate time range
        if analysis_periods:
            time_start = min(p[0] for p in analysis_periods)
            time_end = max(p[0] + p[1] for p in analysis_periods)
            time_range_str = f"{time_start:.1f}s - {time_end:.1f}s"
        else:
            time_range_str = "N/A"
        
        logger.info(f"\n  === BRNG Frame Analysis Summary ===")
        logger.info(f"  Analyzed {len(violations)} frames across {len(analysis_periods)} periods ({time_range_str})\n")
        
        # Aggregate diagnostic types from all violations
        diagnostic_counts = {}
        edge_artifact_edges = set()
        
        for v in violations:
            if v.diagnostics:
                for diag in v.diagnostics:
                    # Normalize edge artifact messages
                    if diag.startswith("Edge artifacts"):
                        diagnostic_counts["Edge artifacts"] = diagnostic_counts.get("Edge artifacts", 0) + 1
                        # Extract edge names from message like "Edge artifacts (left, right)"
                        if "(" in diag and ")" in diag:
                            edges_str = diag[diag.find("(")+1:diag.find(")")]
                            for edge in edges_str.split(", "):
                                edge_artifact_edges.add(edge.strip())
                    elif diag == "Border adjustment recommended":
                        diagnostic_counts["Border adjustment flags"] = diagnostic_counts.get("Border adjustment flags", 0) + 1
                    else:
                        diagnostic_counts[diag] = diagnostic_counts.get(diag, 0) + 1
        
        # Log diagnostic types
        logger.info("  Violation Types Detected:")
        total_violations = len(violations)
        
        # Order diagnostics by relevance
        priority_order = ["Sub-black detected", "Highlight clipping", "Edge artifacts", 
                        "Linear blanking patterns", "Border adjustment flags", 
                        "General broadcast range violations"]
        
        logged_any = False
        for diag_type in priority_order:
            if diag_type in diagnostic_counts:
                count = diagnostic_counts[diag_type]
                pct = (count / total_violations) * 100
                
                if diag_type == "Edge artifacts" and edge_artifact_edges:
                    edges_str = ", ".join(sorted(edge_artifact_edges))
                    logger.info(f"    • {diag_type} ({edges_str}): {count} frames ({pct:.1f}%)")
                else:
                    logger.info(f"    • {diag_type}: {count} frames ({pct:.1f}%)")
                logged_any = True
        
        # Log any remaining diagnostics not in priority order
        for diag_type, count in diagnostic_counts.items():
            if diag_type not in priority_order:
                pct = (count / total_violations) * 100
                logger.info(f"    • {diag_type}: {count} frames ({pct:.1f}%)")
                logged_any = True
        
        if not logged_any:
            logger.info("    • No specific diagnostic patterns identified")
        
        # Add warning if edge percentage is high
        edge_pct = aggregate.get('edge_violation_percentage', 0)
        if edge_pct > 50:
            logger.info(f"    ⚠ High edge percentage ({edge_pct:.1f}%) suggests border detection needs adjustment")
        
       # Log violation distribution statistics
        logger.info(f"\n  Violation Statistics:")
        logger.info(f"    Average BRNG: {stats.get('average_violation_percentage', 0):.2f}%")
        logger.info(f"    Maximum BRNG: {stats.get('max_violation_percentage', 0):.2f}%")
        logger.info(f"    Edge violations: {edge_pct:.1f}% of analyzed frames")
        
        linear_pct = aggregate.get('linear_pattern_percentage', 0)
        if linear_pct > 0:
            logger.info(f"    Linear patterns: {linear_pct:.1f}% of analyzed frames")
        
        logger.info("")  # Blank line for spacing


    def _log_analysis_correlation(self, signalstats_results: SignalstatsResult,
                                brng_results: BRNGAnalysisResult) -> None:
        """Log the correlation between signalstats and BRNG analysis results."""
        if not signalstats_results or not brng_results:
            return
        
        logger.info("  === Analysis Correlation ===\n")
        
        # Signalstats summary
        logger.info("  Signalstats (quantitative full-frame vs active-area comparison):")
        logger.info(f"    Active area violations: {signalstats_results.violation_percentage:.1f}% of frames")
        logger.info(f"    Max BRNG in active area: {signalstats_results.max_brng:.2f}%")
        logger.info(f"    Diagnosis: {signalstats_results.diagnosis}\n")
        
        # BRNG analysis summary
        aggregate = brng_results.aggregate_patterns
        edge_pct = aggregate.get('edge_violation_percentage', 0)
        
        # Determine dominant diagnostic from violations
        diagnostic_counts = {}
        for v in brng_results.violations:
            if v.diagnostics:
                for diag in v.diagnostics:
                    if diag.startswith("Edge artifacts"):
                        diag = "Edge artifacts"
                    elif diag == "Border adjustment recommended":
                        continue  # Skip meta-diagnostics
                    diagnostic_counts[diag] = diagnostic_counts.get(diag, 0) + 1
        
        dominant_diag = max(diagnostic_counts.items(), key=lambda x: x[1])[0] if diagnostic_counts else "Unknown"
        
        logger.info("  BRNG Analysis (qualitative frame inspection):")
        logger.info(f"    Edge violation ratio: {edge_pct:.1f}%")
        logger.info(f"    Dominant diagnostic: {dominant_diag}\n")
        
        # Interpretation
        logger.info("  Interpretation:")
        
        # Determine agreement between methods
        signalstats_says_border = "border" in signalstats_results.diagnosis.lower()
        signalstats_says_content = "active" in signalstats_results.diagnosis.lower() and "requires" in signalstats_results.diagnosis.lower()
        brng_says_border = brng_results.requires_border_adjustment or edge_pct > 50
        brng_says_content = not brng_says_border and edge_pct < 30
        
        if signalstats_says_border and brng_says_border:
            logger.info("    ✓ Both methods agree: violations are concentrated at frame edges")
            logger.info("      → Border detection likely missed some blanking areas")
            logger.info("      → Active picture content appears broadcast-safe once borders are corrected\n")
        elif signalstats_says_content and brng_says_content:
            logger.info("    ✓ Both methods agree: violations are in the active picture area")
            logger.info("      → Content itself has broadcast range issues")
            logger.info("      → Review source material or encoding parameters\n")
        elif signalstats_says_content and brng_says_border:
            logger.info("    ⚠ Methods show mixed results:")
            logger.info("      → Signalstats: active area has violations")
            logger.info("      → BRNG analysis: high edge violation percentage")
            logger.info("      → Both content issues and border detection may need attention\n")
        elif signalstats_says_border and brng_says_content:
            logger.info("    ⚠ Methods show mixed results:")
            logger.info("      → Signalstats: border areas have more violations")
            logger.info("      → BRNG analysis: violations spread throughout frame")
            logger.info("      → Review thumbnails to determine actual issue location\n")
        else:
            # Default case
            if edge_pct > 50:
                logger.info("    → High edge violation percentage suggests border issues")
            elif edge_pct < 20:
                logger.info("    → Low edge percentage suggests content-based violations")
            else:
                logger.info("    → Mixed violation distribution - review thumbnails for details")
            logger.info("")
    
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
        
        logger.info(f"Results saved to: {output_file}\n")
        print(results['summary'])

    def _analyze_qctools_violation_distribution(self, violations: List[FrameViolation]) -> List[Tuple[float, int]]:
        """
        Analyze the temporal distribution of QCTools violations and suggest analysis periods.
        
        Returns:
            List of (start_time, duration) tuples for suggested periods
        """
        if not violations:
            logger.info("  No QCTools violations to analyze distribution")
            return []
        
        # Get all violation timestamps
        timestamps = [v.timestamp for v in violations]
        timestamps.sort()
        
        # Log the overall distribution
        logger.info(f"\n  === QCTools Violation Distribution ===")
        logger.debug(f"  Total violations found: {len(violations)}")
        logger.debug(f"  Time range: {timestamps[0]:.1f}s - {timestamps[-1]:.1f}s")
        
        # Create a histogram of violations over time (10-second bins)
        bin_size = 10.0
        num_bins = int((timestamps[-1] - timestamps[0]) / bin_size) + 1
        bins = [0] * num_bins
        
        for ts in timestamps:
            bin_idx = int((ts - timestamps[0]) / bin_size)
            if 0 <= bin_idx < num_bins:
                bins[bin_idx] += 1
        
        # Find the top periods with most violations
        bin_scores = []
        for i, count in enumerate(bins):
            if count > 0:
                start_time = timestamps[0] + i * bin_size
                bin_scores.append((start_time, count))
        
        # Sort by violation count
        bin_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Log the top bins
        logger.debug(f"  Top 10-second bins with violations:")
        for i, (start_time, count) in enumerate(bin_scores[:10]):
            logger.debug(f"    {i+1}. {start_time:.1f}s - {start_time+bin_size:.1f}s: {count} violations")
        
        # Select 3 periods based on violation density
        suggested_periods = []
        used_ranges = []
        
        for start_time, count in bin_scores:
            # Check if this period overlaps with any already selected
            overlaps = False
            for used_start, used_end in used_ranges:
                if not (start_time + 60 < used_start or start_time > used_end):
                    overlaps = True
                    break
            
            if not overlaps:
                suggested_periods.append((start_time, 60))  # 60-second periods
                used_ranges.append((start_time, start_time + 60))
                
                if len(suggested_periods) >= 3:
                    break
        
        # If we couldn't find 3 non-overlapping dense regions, fall back to even distribution
        if len(suggested_periods) < 3:
            logger.debug(f"  Only found {len(suggested_periods)} dense violation regions")
            # Add evenly distributed periods from the violation range
            violation_duration = timestamps[-1] - timestamps[0]
            if violation_duration > 180:  # If violations span > 3 minutes
                spacing = violation_duration / 3
                for i in range(3 - len(suggested_periods)):
                    start = timestamps[0] + (i + len(suggested_periods)) * spacing
                    suggested_periods.append((start, 60))
        
        logger.debug(f"\n  QCTools-based suggested analysis periods:")
        for i, (start, duration) in enumerate(suggested_periods):
            logger.debug(f"    Period {i+1}: {start:.1f}s - {start+duration:.1f}s")
        
        return suggested_periods


def analyze_frame_quality(video_path: str, 
                         border_data_path: str = None,
                         output_dir: str = None,
                         frame_config: 'FrameAnalysisConfig' = None,
                         color_bars_end_time: float = None,
                         signals = None) -> Dict:
    """
    Main entry point for frame analysis from processing_mgmt.
    
    Args:
        video_path: Path to video file
        border_data_path: Optional path to existing border data JSON
        output_dir: Output directory for results
        frame_config: FrameAnalysisConfig dataclass with analysis parameters
        color_bars_end_time: End time of color bars if detected
        signals: Optional signals object for emitting step completion events
    
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
    skip_color_bars = bool(frame_config.brng_skip_color_bars)
    max_refinements = frame_config.max_border_retries
    
    # Run enhanced analysis
    analyzer = EnhancedFrameAnalysis(video_path, output_dir, signals=signals)
    
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
        max_refinement_iterations=max_refinements,
        color_bars_end_time=color_bars_end_time,
        signals=signals
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