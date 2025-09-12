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

# Data classes for structured results
@dataclass
class FrameViolation:
    """Represents a frame with BRNG violations"""
    frame_num: int
    timestamp: float
    brng_low: float
    brng_high: float
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
    
    def parse_for_violations_streaming(self, max_frames: int = 100, 
                                      skip_color_bars: bool = True) -> List[FrameViolation]:
        """Stream parse QCTools report for BRNG violations"""
        violations = []
        chunk_size = 1000
        color_bars_end_frame = 0

        # Counters
        total_frames_checked = 0
        frames_with_violations = 0
        max_brng_low = 0
        max_brng_high = 0
        
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
                    frame_num = int(elem.get('n', 0))
                    total_frames_checked += 1
                    
                    # Skip color bars if requested
                    if skip_color_bars and frame_num < color_bars_end_frame:
                        elem.clear()
                        root.clear()
                        continue
                    
                    # Extract frame data
                    frame_data = self._extract_frame_violations(elem, frame_num)
                    if frame_data:
                        frames_with_violations += 1
                        max_brng_low = max(max_brng_low, frame_data.brng_low)
                        max_brng_high = max(max_brng_high, frame_data.brng_high)
                        frame_buffer.append(frame_data)
                    
                    elem.clear()
                    root.clear()
                    
                    # Process buffer when full
                    if len(frame_buffer) >= chunk_size:
                        violations.extend(self._process_violation_buffer(frame_buffer))
                        frame_buffer = []
                        
                        # Keep only top violations to manage memory
                        if len(violations) > max_frames * 2:
                            violations.sort(key=lambda x: x.violation_score, reverse=True)
                            violations = violations[:max_frames]
            
            # Process remaining buffer
            if frame_buffer:
                violations.extend(self._process_violation_buffer(frame_buffer))
            
            file_handle.close()
            
            # Log summary after parsing
            logger.info(f"  Checked {total_frames_checked:,} frames from QCTools report")
            logger.info(f"  Found {frames_with_violations:,} frames with BRNG violations ({frames_with_violations/total_frames_checked*100:.1f}%)")
            if frames_with_violations > 0:
                logger.info(f"  Max BRNG values - Low: {max_brng_low:.4f}%, High: {max_brng_high:.4f}%")
            
        except Exception as e:
            logger.error(f"Error parsing QCTools report: {e}")
        
        violations.sort(key=lambda x: x.violation_score, reverse=True)
        return violations[:max_frames]
    
    def _extract_frame_violations(self, elem, frame_num: int) -> Optional[FrameViolation]:
        """Extract violation data from frame element"""
        try:
            brng_low = float(elem.findtext('.//tag[@key="lavfi.signalstats.BRNG_low"]', '0'))
            brng_high = float(elem.findtext('.//tag[@key="lavfi.signalstats.BRNG_high"]', '0'))
            
            score = abs(brng_low) + abs(brng_high)
            if score > 0.01:
                return FrameViolation(
                    frame_num=frame_num,
                    timestamp=frame_num / self.fps,
                    brng_low=brng_low,
                    brng_high=brng_high,
                    violation_score=score
                )
        except:
            pass
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
                                           skip_start_seconds: float = 0) -> BRNGAnalysisResult:
        """
        Perform differential BRNG detection by creating highlighted and original versions.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Add logging
        logger.info(f"  Creating temporary comparison videos (duration: {duration_limit}s)")
        logger.info(f"  Output directory: {output_dir}")
        
        # Create temporary directory for processing
        temp_dir = output_dir / "temp_brng"
        temp_dir.mkdir(exist_ok=True)
        
        # Generate highlighted and original videos
        highlighted_path = temp_dir / f"{self.video_path.stem}_highlighted.mp4"
        original_path = temp_dir / f"{self.video_path.stem}_original.mp4"
        
        if not self._create_comparison_videos(highlighted_path, original_path, 
                                             duration_limit, skip_start_seconds):
            logger.error("Failed to create comparison videos")
            return None
        
        # Analyze violations
        violations = self._analyze_differential_violations(
            highlighted_path, original_path, skip_start_seconds
        )

        logger.info(f"  Analyzed {len(violations)} frames with potential violations")
        
        # Analyze patterns and generate report
        aggregate_patterns = self._analyze_aggregate_patterns(violations)
        actionable_report = self._generate_actionable_report(violations, aggregate_patterns)
        
        # Create diagnostic thumbnails for worst violations
        thumbnails = []
        if violations and len(violations) > 0:
            thumb_dir = output_dir / "brng_thumbnails"
            thumb_dir.mkdir(exist_ok=True)
            logger.info(f"  Creating diagnostic thumbnails for top {min(5, len(violations))} violations")
            thumbnails = self._create_diagnostic_thumbnails(violations[:5], highlighted_path, original_path, thumb_dir)
            logger.info(f"  Saved {len(thumbnails)} thumbnails to {thumb_dir}")
        
        # Clean up temp files
        try:
            highlighted_path.unlink()
            original_path.unlink()
            temp_dir.rmdir()
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
    
    def _create_comparison_videos(self, highlighted_path: Path, original_path: Path,
                                 duration_limit: int, skip_start: float) -> bool:
        """Create highlighted and original versions for differential analysis"""
        # Build crop filter if active area exists
        crop_filter = ""
        if self.active_area:
            x, y, w, h = self.active_area
            crop_filter = f"crop={w}:{h}:{x}:{y},"
        
        # Seek arguments
        seek_args = ["-ss", str(skip_start)] if skip_start > 0 else []
        
        # Create highlighted version
        highlighted_cmd = [
            "ffmpeg",
            *seek_args,
            "-i", str(self.video_path),
            "-t", str(duration_limit),
            "-vf", f"{crop_filter}signalstats=out=brng:color=cyan",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            str(highlighted_path)
        ]
        
        # Create original version
        original_cmd = [
            "ffmpeg",
            *seek_args,
            "-i", str(self.video_path),
            "-t", str(duration_limit),
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
                                        skip_offset: float) -> List[FrameViolation]:
        """Analyze violations using differential detection"""
        violations = []
        
        cap_h = cv2.VideoCapture(str(highlighted_path))
        cap_o = cv2.VideoCapture(str(original_path))
        
        # Sample frames adaptively
        total_frames = int(cap_h.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, total_frames - 1, min(100, total_frames), dtype=int)
        
        for idx in sample_indices:
            cap_h.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_o.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            ret_h, frame_h = cap_h.read()
            ret_o, frame_o = cap_o.read()
            
            if not ret_h or not ret_o:
                continue
            
            # Detect violations differentially
            violation_mask = self._detect_differential_violations_frame(frame_h, frame_o)
            violation_pixels = int(np.sum(violation_mask > 0))
            
            if violation_pixels > 0:
                # Analyze patterns
                pattern_analysis = self._analyze_violation_patterns(violation_mask, frame_o)
                
                # Adjust timestamp to account for skip
                actual_timestamp = (idx / self.fps) + skip_offset
                
                violation = FrameViolation(
                    frame_num=int(idx + (skip_offset * self.fps)),
                    timestamp=actual_timestamp,
                    brng_low=0,  # Will be updated if we have QCTools data
                    brng_high=0,
                    violation_score=violation_pixels / (self.width * self.height),
                    violation_pixels=violation_pixels,
                    violation_percentage=(violation_pixels / (self.width * self.height)) * 100,
                    diagnostics=pattern_analysis.get('diagnostics', []),
                    pattern_analysis=pattern_analysis
                )
                violations.append(violation)
        
        cap_h.release()
        cap_o.release()
        
        # Sort by violation score
        violations.sort(key=lambda x: x.violation_score, reverse=True)
        return violations
    
    def _detect_differential_violations_frame(self, highlighted: np.ndarray, 
                                             original: np.ndarray) -> np.ndarray:
        """Detect violations by comparing highlighted vs original frame"""
        # Calculate difference
        diff = cv2.absdiff(highlighted, original)
        
        # Convert to HSV to isolate cyan changes
        diff_hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
        
        # Detect cyan (BRNG highlight color)
        cyan_mask = cv2.inRange(diff_hsv,
                               np.array([80, 30, 30]),
                               np.array([110, 255, 255]))
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
        
        return cyan_mask
    
    def _analyze_violation_patterns(self, violation_mask: np.ndarray, 
                                   frame: np.ndarray) -> Dict:
        """Analyze patterns in violations for diagnostics"""
        h, w = violation_mask.shape
        
        # Detect edge violations
        edge_info = self._detect_edge_violations(violation_mask)
        
        # Generate diagnostics
        diagnostics = []
        
        if edge_info['has_edge_violations']:
            if edge_info.get('linear_patterns'):
                diagnostics.append("Linear blanking patterns detected")
            elif edge_info.get('continuous_edges'):
                edges_str = ', '.join(edge_info['edges_affected'])
                diagnostics.append(f"Continuous edge artifacts ({edges_str})")
            else:
                edges_str = ', '.join(edge_info['edges_affected'])
                diagnostics.append(f"Edge artifacts ({edges_str})")
        
        # Check for general violations
        if not edge_info['has_edge_violations']:
            diagnostics.append("General broadcast range violations")
        
        return {
            'edge_violations': edge_info,
            'diagnostics': diagnostics,
            'has_boundary_artifacts': edge_info['has_edge_violations'],
            'severity': edge_info.get('severity', 'low')
        }
    
    def _detect_edge_violations(self, violation_mask: np.ndarray, 
                               edge_width: int = 15) -> Dict:
        """Enhanced edge violation detection"""
        h, w = violation_mask.shape
        edge_info = {
            'has_edge_violations': False,
            'edges_affected': [],
            'linear_patterns': {},
            'continuous_edges': [],
            'severity': 'none',
            'expansion_recommendations': {}
        }
        
        edges_to_check = [
            ('left', violation_mask[:, :edge_width], 'vertical'),
            ('right', violation_mask[:, -edge_width:], 'vertical'),
            ('top', violation_mask[:edge_width, :], 'horizontal'),
            ('bottom', violation_mask[-edge_width:, :], 'horizontal')
        ]
        
        for edge_name, edge_region, orientation in edges_to_check:
            if edge_region.size == 0:
                continue
            
            violation_pct = (np.sum(edge_region > 0) / edge_region.size) * 100
            
            # Detect linear patterns
            linear_score = 0
            if orientation == 'vertical':
                for row in range(edge_region.shape[0]):
                    if np.any(edge_region[row, :] > 0):
                        linear_score += 1
                linear_pct = (linear_score / edge_region.shape[0]) * 100
            else:
                for col in range(edge_region.shape[1]):
                    if np.any(edge_region[:, col] > 0):
                        linear_score += 1
                linear_pct = (linear_score / edge_region.shape[1]) * 100
            
            edge_info['linear_patterns'][edge_name] = linear_pct
            
            # Determine if this edge has violations
            if violation_pct > 5 or linear_pct > 30:
                edge_info['edges_affected'].append(edge_name)
                edge_info['has_edge_violations'] = True
                
                if linear_pct > 50:
                    edge_info['continuous_edges'].append(edge_name)
                    edge_info['expansion_recommendations'][edge_name] = 10
        
        # Determine severity
        if len(edge_info['continuous_edges']) >= 2:
            edge_info['severity'] = 'high'
        elif len(edge_info['continuous_edges']) >= 1:
            edge_info['severity'] = 'medium'
        elif len(edge_info['edges_affected']) >= 1:
            edge_info['severity'] = 'low'
        
        return edge_info
    
    def _analyze_aggregate_patterns(self, violations: List[FrameViolation]) -> Dict:
        """Analyze aggregate patterns across all violations"""
        if not violations:
            return {
                'requires_border_adjustment': False,
                'edge_violation_percentage': 0,
                'continuous_edge_percentage': 0
            }
        
        # Count edge violations
        edge_violations = 0
        continuous_edges = 0
        all_affected_edges = []
        
        for v in violations:
            if v.pattern_analysis:
                edge_info = v.pattern_analysis.get('edge_violations', {})
                if edge_info.get('has_edge_violations'):
                    edge_violations += 1
                    all_affected_edges.extend(edge_info.get('edges_affected', []))
                if edge_info.get('continuous_edges'):
                    continuous_edges += 1
        
        edge_violation_pct = (edge_violations / len(violations)) * 100
        continuous_edge_pct = (continuous_edges / len(violations)) * 100
        
        # Determine if border adjustment is needed
        requires_adjustment = (
            continuous_edge_pct > 15 or
            edge_violation_pct > 30 or
            (continuous_edge_pct > 10 and len(set(all_affected_edges)) >= 2)
        )
        
        # Calculate expansion recommendations
        expansion_recs = {}
        if requires_adjustment:
            edge_counts = defaultdict(int)
            for edge in all_affected_edges:
                edge_counts[edge] += 1
            
            for edge, count in edge_counts.items():
                if count > len(violations) * 0.2:
                    expansion_recs[edge] = 10 if count > len(violations) * 0.5 else 5
        
        return {
            'requires_border_adjustment': requires_adjustment,
            'edge_violation_percentage': edge_violation_pct,
            'continuous_edge_percentage': continuous_edge_pct,
            'boundary_edges_detected': list(set(all_affected_edges)),
            'expansion_recommendations': expansion_recs
        }
    
    def _generate_actionable_report(self, violations: List[FrameViolation],
                                   aggregate_patterns: Dict) -> Dict:
        """Generate actionable recommendations"""
        if not violations:
            return {
                'overall_assessment': 'No BRNG violations detected',
                'action_priority': 'none',
                'recommendations': []
            }
        
        recommendations = []
        
        if aggregate_patterns['requires_border_adjustment']:
            recommendations.append({
                'issue': 'Border Detection Needs Adjustment',
                'severity': 'high',
                'action': 'Re-run border detection with adjusted parameters'
            })
            action_priority = 'high'
        else:
            avg_violation_pct = np.mean([v.violation_percentage for v in violations])
            if avg_violation_pct > 1.0:
                recommendations.append({
                    'issue': 'Content BRNG Violations',
                    'severity': 'medium',
                    'action': 'Review source material or encoding parameters'
                })
                action_priority = 'medium'
            else:
                action_priority = 'low'
        
        return {
            'overall_assessment': self._get_overall_assessment(violations, aggregate_patterns),
            'action_priority': action_priority,
            'recommendations': recommendations
        }
    
    def _get_overall_assessment(self, violations: List[FrameViolation],
                               aggregate_patterns: Dict) -> str:
        """Generate overall assessment text"""
        if not violations:
            return "Video is broadcast-safe with no BRNG violations"
        
        if aggregate_patterns['requires_border_adjustment']:
            return "Border detection adjustment required - edges contain violations"
        
        avg_violation_pct = np.mean([v.violation_percentage for v in violations])
        if avg_violation_pct < 0.1:
            return "Video appears broadcast-compliant with minimal violations"
        elif avg_violation_pct < 1.0:
            return "Minor broadcast range issues detected"
        else:
            return "Broadcast range issues requiring attention"
    
    def _create_diagnostic_thumbnails(self, violations: List[FrameViolation],
                                     highlighted_path: Path,
                                     original_path: Path,
                                     output_dir: Path) -> List[str]:
        """Create 4-quadrant diagnostic thumbnails"""
        thumbnails = []
        
        cap_h = cv2.VideoCapture(str(highlighted_path))
        cap_o = cv2.VideoCapture(str(original_path))
        
        for i, violation in enumerate(violations[:5]):
            # Adjust for the fact we're using the processed video
            frame_idx = int((violation.timestamp - (violation.timestamp // self.duration) * self.duration) * self.fps)
            
            cap_h.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap_o.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret_h, frame_h = cap_h.read()
            ret_o, frame_o = cap_o.read()
            
            if not ret_h or not ret_o:
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
            
            # Bottom-left: Violations only
            violation_mask = self._detect_differential_violations_frame(frame_h, frame_o)
            violations_only = np.zeros_like(frame_o)
            violations_only[violation_mask > 0] = [255, 255, 0]  # Cyan
            viz[h+padding:h*2+padding, 0:w] = violations_only
            
            # Bottom-right: Analysis info
            info_panel = np.zeros((h, w, 3), dtype=np.uint8)
            self._add_info_text(info_panel, violation)
            viz[h+padding:h*2+padding, w+padding:w*2+padding] = info_panel
            
            # Save thumbnail
            thumb_filename = f"{self.video_path.stem}_brng_{i:03d}.jpg"
            thumb_path = output_dir / thumb_filename
            cv2.imwrite(str(thumb_path), viz)
            thumbnails.append(str(thumb_path))
        
        cap_h.release()
        cap_o.release()
        
        return thumbnails
    
    def _add_info_text(self, panel: np.ndarray, violation: FrameViolation):
        """Add analysis info text to panel"""
        h, w = panel.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        lines = [
            f"Frame: {violation.frame_num}",
            f"Time: {violation.timestamp:.2f}s",
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
        Analyze using signalstats, with QCTools data when available.
        """
        # Determine analysis periods
        analysis_periods = self._find_analysis_periods(
            content_start_time, 
            color_bars_end_time,
            analysis_duration,
            num_periods,
            border_data.quality_frame_hints if border_data else None
        )
        
        # Determine active area
        active_area = border_data.active_area if border_data else None
        
        # Analyze periods
        all_results = []
        used_qctools = False
        
        for start_time, duration in analysis_periods:
            # Try QCTools first if available
            if self.qctools_report:
                qctools_result = self._parse_qctools_brng(start_time, start_time + duration)
                if qctools_result and self._should_use_qctools(qctools_result):
                    all_results.append(qctools_result)
                    used_qctools = True
                    continue
            
            # Fall back to FFprobe
            ffprobe_result = self._analyze_with_ffprobe(
                active_area, start_time, duration
            )
            if ffprobe_result:
                all_results.append(ffprobe_result)
        
        # Aggregate results
        if not all_results:
            return SignalstatsResult(
                violation_percentage=0,
                max_brng=0,
                avg_brng=0,
                analysis_periods=[],
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
        
        # Generate diagnosis
        diagnosis = self._generate_diagnosis(violation_pct, max_brng)
        
        return SignalstatsResult(
            violation_percentage=violation_pct,
            max_brng=max_brng,
            avg_brng=avg_brng,
            analysis_periods=analysis_periods,
            diagnosis=diagnosis,
            used_qctools=used_qctools
        )
    
    def _find_analysis_periods(self, content_start: float, color_bars_end: float,
                              duration: int, num_periods: int,
                              quality_hints: List[Tuple[float, float]] = None) -> List[Tuple[float, int]]:
        """Find good analysis periods, using quality hints if available"""
        effective_start = max(content_start, color_bars_end or 0) + 10
        
        # Use quality hints if available
        if quality_hints and len(quality_hints) >= num_periods:
            periods = []
            for time_hint, _ in quality_hints[:num_periods]:
                if time_hint >= effective_start:
                    period_start = max(effective_start, time_hint - duration/2)
                    if period_start + duration <= self.duration - 30:
                        periods.append((period_start, duration))
            if len(periods) >= num_periods:
                return periods
        
        # Fall back to even distribution
        available_duration = self.duration - effective_start - 30
        if available_duration < duration:
            return [(effective_start, min(duration, available_duration))]
        
        periods = []
        spacing = (available_duration - duration) / max(1, num_periods - 1)
        for i in range(num_periods):
            start = effective_start + i * spacing
            periods.append((start, duration))
        
        return periods
    
    def _parse_qctools_brng(self, start_time: float, end_time: float) -> Optional[Dict]:
        """Parse BRNG values from QCTools report"""
        if not self.qctools_report:
            return None
        
        parser = QCToolsParser(self.qctools_report, self.fps)
        violations = parser.parse_for_violations_streaming(max_frames=1000)
        
        # Filter to time range
        relevant = [v for v in violations 
                   if start_time <= v.timestamp <= end_time]
        
        if not relevant:
            return None
        
        return {
            'frames_analyzed': len(relevant),
            'frames_with_violations': len([v for v in relevant if v.violation_score > 0]),
            'brng_values': [v.violation_score for v in relevant],
            'source': 'qctools'
        }
    
    def _should_use_qctools(self, qctools_result: Dict) -> bool:
        """Decide if QCTools data is sufficient"""
        if not qctools_result:
            return False
        
        violation_pct = (qctools_result['frames_with_violations'] / 
                        qctools_result['frames_analyzed'] * 100)
        max_brng = max(qctools_result['brng_values']) if qctools_result['brng_values'] else 0
        
        # Use QCTools if violations are minimal
        return violation_pct < 5 and max_brng < 0.05
    
    def _analyze_with_ffprobe(self, active_area: Tuple, 
                             start_time: float, duration: int) -> Dict:
        """Analyze using FFprobe signalstats"""
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
            
            return {
                'frames_analyzed': len(brng_values),
                'frames_with_violations': frames_with_violations,
                'brng_values': brng_values,
                'source': 'ffprobe'
            }
        except Exception as e:
            logger.error(f"FFprobe error: {e}")
            return None
    
    def _generate_diagnosis(self, violation_pct: float, max_brng: float) -> str:
        """Generate diagnosis based on results"""
        if violation_pct < 10 and max_brng < 0.01:
            return "Video appears broadcast-compliant"
        elif violation_pct < 50 and max_brng < 0.1:
            return "Minor BRNG violations - likely acceptable"
        elif max_brng > 2.0:
            return "Significant BRNG violations requiring correction"
        else:
            return "Moderate BRNG violations detected"


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
        
        # Initialize components
        self.border_detector = SophisticatedBorderDetector(video_path)
        self.brng_analyzer = None  # Will be initialized with border data
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
        
        # Step 1: Parse QCTools for initial violations (if available)
        violations = []
        frames_with_qctools_violations = 0
        color_bars_end_time = 0
        
        if self.qctools_parser:
            logger.info("Parsing QCTools report for violations...")
            violations = self.qctools_parser.parse_for_violations_streaming(
                max_frames=100,
                skip_color_bars=skip_color_bars
            )
            frames_with_qctools_violations = len(violations)
            
            # Fix the confusing "0 violations" issue
            if frames_with_qctools_violations == 0:
                results['qctools_violations_found'] = "No BRNG violations detected"
            else:
                results['qctools_violations_found'] = frames_with_qctools_violations
            
            # Detect color bars duration if needed
            if skip_color_bars:
                color_bars_end_time = self._detect_color_bars_duration()
                results['color_bars_end_time'] = color_bars_end_time
        else:
            logger.info("No QCTools report found, proceeding with direct analysis")
        
        # Step 2: Initial border detection
        logger.info(f"Detecting borders using {method} method...")
        border_results = self.border_detector.detect_borders_with_quality_assessment(
            violations=violations,
            method=method
        )
        results['initial_borders'] = asdict(border_results)
        
        # Step 3: BRNG analysis with differential detection
        logger.info("Analyzing BRNG violations...")
        self.brng_analyzer = DifferentialBRNGAnalyzer(self.video_path, border_results)
        
        brng_results = self.brng_analyzer.analyze_with_differential_detection(
            output_dir=self.output_dir,
            duration_limit=duration_limit,
            skip_start_seconds=color_bars_end_time
        )
        results['brng_analysis'] = asdict(brng_results) if brng_results else None
        
        # Step 4: Iterative border refinement (if needed and using sophisticated method)
        refinement_iterations = 0
        if method == 'sophisticated' and brng_results and brng_results.requires_border_adjustment:
            logger.info("Border refinement needed - detected BRNG violations at frame edges")
            logger.info("  This suggests the initial border detection may have missed blanking/inactive areas")
            
            edge_pct = brng_results.aggregate_patterns.get('edge_violation_percentage', 0)
            continuous_pct = brng_results.aggregate_patterns.get('continuous_edge_percentage', 0)
            logger.info(f"  Edge violations: {edge_pct:.1f}% of analyzed frames")
            logger.info(f"  Continuous edge patterns: {continuous_pct:.1f}% of analyzed frames")
            
            while (refinement_iterations < max_refinement_iterations and 
                   brng_results.requires_border_adjustment):
                
                refinement_iterations += 1
                logger.info(f"Refinement iteration {refinement_iterations}/{max_refinement_iterations}:")

                # Log what we're adjusting
                if brng_results.refinement_recommendations:
                    adjustments = []
                    for edge, pixels in brng_results.refinement_recommendations.items():
                        adjustments.append(f"{edge}:{pixels}px")
                    logger.info(f"  Expanding borders: {', '.join(adjustments)}")

                # After refinement
                previous_area = border_results.active_area
                border_results = self.border_detector.refine_borders(border_results, brng_results)
                new_area = border_results.active_area

                logger.info(f"  Active area: {previous_area[2]}x{previous_area[3]} → {new_area[2]}x{new_area[3]}")
                logger.info(f"  Border change: {abs(new_area[2]-previous_area[2])}x{abs(new_area[3]-previous_area[3])} pixels")
                
                # Log the change
                new_area = border_results.active_area
                logger.info(f"  Active area adjusted from {previous_area[2]}x{previous_area[3]} to {new_area[2]}x{new_area[3]}")
                
                # Re-analyze with new borders
                self.brng_analyzer = DifferentialBRNGAnalyzer(self.video_path, border_results)
                previous_brng = brng_results
                
                brng_results = self.brng_analyzer.analyze_with_differential_detection(
                    output_dir=self.output_dir,
                    duration_limit=duration_limit,
                    skip_start_seconds=color_bars_end_time
                )
                
                # Check for improvement
                improved = self._is_meaningful_improvement(previous_brng, brng_results)
                if not improved:
                    prev_violations = len(previous_brng.violations)
                    curr_violations = len(brng_results.violations)
                    
                    if prev_violations > 0 and curr_violations > 0:
                        prev_worst = previous_brng.violations[0].violation_percentage
                        curr_worst = brng_results.violations[0].violation_percentage
                        
                        logger.info(f"  Refinement complete - minimal improvement detected")
                        logger.info(f"    Violations: {prev_violations} → {curr_violations} frames")
                        logger.info(f"    Worst violation: {prev_worst:.4f}% → {curr_worst:.4f}%")
                        logger.info(f"    (Need 20% reduction for meaningful improvement)")
                    else:
                        logger.info(f"  Refinement complete - violations remain in content area, not edges")
                    
                    logger.info("  Stopping refinement - further border adjustments unlikely to help")
                    break
                else:
                    logger.info(f"  Refinement improved results, continuing...")
            
            results['refinement_iterations'] = refinement_iterations
            results['final_borders'] = asdict(border_results)
            results['final_brng_analysis'] = asdict(brng_results) if brng_results else None
        elif method == 'sophisticated' and brng_results and not brng_results.requires_border_adjustment:
            logger.info("  No border refinement needed - violations are in content, not edges")
        
        # Step 5: Signalstats analysis
        logger.info("Running signalstats analysis...")
        signalstats_results = self.signalstats_analyzer.analyze_with_signalstats(
            border_data=border_results,
            content_start_time=0,
            color_bars_end_time=color_bars_end_time,
            analysis_duration=60,
            num_periods=3
        )
        results['signalstats'] = asdict(signalstats_results)
        
        # Step 6: Generate comprehensive summary
        results['summary'] = self._generate_summary(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _detect_color_bars_duration(self) -> float:
        """Detect color bars duration from video or QCTools report"""
        # Simplified color bars detection
        # In production, would use the full detectBars logic from qct_parse
        return 0  # Placeholder
    
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
        output_file = self.output_dir / f"{self.video_id}_enhanced_frame_analysis.json"
        
        # Convert any remaining non-serializable objects
        def serialize(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=serialize)
        
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