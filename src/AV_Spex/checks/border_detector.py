#!/usr/bin/env python3
"""
Video Border Detection Script with Enhanced Frame Quality Assessment

Detects blanking borders and active picture areas in video files using OpenCV.
Enhanced with robust frame quality assessment to filter out black frames,
title cards, static content, and other non-video frames.
"""

import cv2
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from AV_Spex.utils.log_setup import logger

def detect_simple_borders(video_path, border_size=25, output_dir=None):
        """
        Simple border detection that assumes a fixed border size on all edges.
        This is the "dumb" version that doesn't analyze content.
        
        Args:
            video_path (str): Path to the video file
            border_size (int): Number of pixels to crop from each edge (default: 25)
            output_dir (str, optional): Directory to save border data JSON
            
        Returns:
            dict: Border detection results with active area
        """
        
        video_path = Path(video_path)
        video_name = video_path.stem
        
        # Open video to get dimensions
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        # Calculate active area (simple fixed borders)
        active_x = border_size
        active_y = border_size
        active_width = width - (2 * border_size)
        active_height = height - (2 * border_size)
        
        # Ensure we don't have negative dimensions
        if active_width <= 0 or active_height <= 0:
            logger.warning(f"Border size {border_size} is too large for video {width}x{height}")
            # Fall back to full frame
            active_x = 0
            active_y = 0
            active_width = width
            active_height = height
            actual_border_size = 0
        else:
            actual_border_size = border_size
        
        # Prepare results in the same format as the sophisticated detector
        results = {
            'video_file': str(video_path),
            'video_properties': {
                'width': width,
                'height': height,
                'fps': float(fps),
                'duration': float(duration),
                'total_frames': total_frames
            },
            'active_area': [active_x, active_y, active_width, active_height],
            'border_regions': {
                'left_border': (0, 0, actual_border_size, height) if actual_border_size > 0 else None,
                'right_border': (width - actual_border_size, 0, actual_border_size, height) if actual_border_size > 0 else None,
                'top_border': (0, 0, width, actual_border_size) if actual_border_size > 0 else None,
                'bottom_border': (0, height - actual_border_size, width, actual_border_size) if actual_border_size > 0 else None
            },
            'detection_method': 'simple_fixed',
            'border_size_used': actual_border_size,
            'head_switching_artifacts': None  # Not detected in simple mode
        }
        
        # Log what we found
        logger.info(f"Simple border detection complete for {video_name}")
        logger.info(f"  Video dimensions: {width}x{height}")
        logger.info(f"  Using fixed border: {actual_border_size}px on all sides")
        logger.info(f"  Active area: {active_width}x{active_height} at ({active_x},{active_y})")
        
        # Save to JSON if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            json_path = output_dir / f"{video_name}_border_data.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"  Border data saved to: {json_path}\n")
        
        return results

class VideoBorderDetector:
    """
    Detects blanking borders and active picture areas in video files
    """
    
    def __init__(self, video_path, sample_frames=30):
        self.video_path = str(video_path)
        self.sample_frames = sample_frames
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        logger.info(f"✓ Video loaded: {self.width}x{self.height}, {self.fps:.2f}fps, {self.duration:.1f}s \n")
    
    
    def assess_frame_quality(self, frame, previous_frame=None, strict_mode=False):
        """
        Comprehensive frame quality assessment to identify suitable video content frames
        
        Args:
            frame: Current frame to assess
            previous_frame: Previous frame for motion detection (optional)
            strict_mode: If True, apply stricter criteria for frame selection
            
        Returns:
            dict: Quality assessment results with scores and flags
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Initialize results
        assessment = {
            'is_suitable': False,
            'scores': {},
            'flags': {},
            'reasons_rejected': []
        }
        
        # 1. Basic brightness analysis
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        assessment['scores']['brightness'] = float(mean_brightness)
        assessment['scores']['contrast'] = float(std_brightness)
        
        # Reject very dark frames (likely black frames, fades)
        if mean_brightness < 15:
            assessment['flags']['too_dark'] = True
            assessment['reasons_rejected'].append(f"Too dark (brightness: {mean_brightness:.1f})")
        
        # Reject very bright frames (likely white frames, overexposed)
        elif mean_brightness > 240:
            assessment['flags']['too_bright'] = True
            assessment['reasons_rejected'].append(f"Too bright (brightness: {mean_brightness:.1f})")
        
        # 2. Contrast analysis - detect low contrast frames (solid colors, fades)
        if strict_mode:
            min_contrast = 25
        else:
            min_contrast = 15
            
        if std_brightness < min_contrast:
            assessment['flags']['low_contrast'] = True
            assessment['reasons_rejected'].append(f"Low contrast (std: {std_brightness:.1f})")
        
        # 3. Detect predominantly uniform frames (solid colors, simple graphics)
        # Calculate what percentage of pixels are within a narrow range
        median_brightness = np.median(gray)
        tolerance = 20
        uniform_pixels = np.sum(np.abs(gray - median_brightness) < tolerance)
        uniform_percentage = (uniform_pixels / gray.size) * 100
        
        assessment['scores']['uniform_percentage'] = float(uniform_percentage)
        
        if uniform_percentage > 85:
            assessment['flags']['too_uniform'] = True
            assessment['reasons_rejected'].append(f"Too uniform ({uniform_percentage:.1f}% similar pixels)")
        
        # 4. Detect potential title cards/text frames using edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        assessment['scores']['edge_density'] = float(edge_density)
        
        # High edge density might indicate text/graphics rather than natural video
        if strict_mode and edge_density > 0.15:
            assessment['flags']['high_edge_density'] = True
            assessment['reasons_rejected'].append(f"High edge density ({edge_density:.3f}) - possible text/graphics")
        elif edge_density > 0.25:
            assessment['flags']['very_high_edge_density'] = True
            assessment['reasons_rejected'].append(f"Very high edge density ({edge_density:.3f}) - likely text/graphics")
        
        # 5. Motion detection (if previous frame available)
        if previous_frame is not None:
            prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.mean(frame_diff)
            
            assessment['scores']['motion'] = float(motion_score)
            
            # Very low motion might indicate static title cards or paused content
            if strict_mode and motion_score < 2:
                assessment['flags']['too_static'] = True
                assessment['reasons_rejected'].append(f"Too static (motion: {motion_score:.1f})")
        
        # 6. Detect letterbox/pillarbox patterns that might indicate title sequences
        # Check if frame has very dark borders that are too large (not just blanking)
        border_threshold = 30
        
        # Check top and bottom rows
        top_brightness = np.mean(gray[:10, :])
        bottom_brightness = np.mean(gray[-10:, :])
        left_brightness = np.mean(gray[:, :10])
        right_brightness = np.mean(gray[:, -10:])
        
        dark_borders = sum([
            top_brightness < border_threshold,
            bottom_brightness < border_threshold,
            left_brightness < border_threshold,
            right_brightness < border_threshold
        ])
        
        assessment['scores']['dark_borders'] = int(dark_borders)
        
        # If all sides are very dark, might be a transition frame
        if dark_borders >= 3:
            assessment['flags']['dark_border_frame'] = True
            assessment['reasons_rejected'].append("Frame has dark borders on multiple sides")
        
        # 7. Histogram analysis - detect unusual distributions
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / np.sum(hist)
        
        # Check for bimodal distributions (might indicate graphics/text)
        # Find peaks in histogram
        hist_smooth = np.convolve(hist_normalized, np.ones(5)/5, mode='same')
        peaks = []
        for i in range(5, len(hist_smooth)-5):
            if (hist_smooth[i] > hist_smooth[i-2] and 
                hist_smooth[i] > hist_smooth[i+2] and 
                hist_smooth[i] > 0.01):  # At least 1% of pixels
                peaks.append(i)
        
        assessment['scores']['histogram_peaks'] = len(peaks)
        
        # Multiple distinct peaks might indicate text on background
        if len(peaks) >= 3 and strict_mode:
            assessment['flags']['multi_peak_histogram'] = True
            assessment['reasons_rejected'].append(f"Multiple histogram peaks ({len(peaks)}) - possible text/graphics")
        
        # 8. Final suitability assessment
        if not assessment['reasons_rejected']:
            assessment['is_suitable'] = True
            
            # Calculate overall quality score (0-1)
            brightness_score = 1.0 - abs(mean_brightness - 120) / 120
            contrast_score = min(std_brightness / 50.0, 1.0)
            uniform_score = max(0, (100 - uniform_percentage) / 100)
            edge_score = max(0, 1.0 - edge_density / 0.2)
            
            overall_score = (brightness_score * 0.3 + 
                           contrast_score * 0.3 + 
                           uniform_score * 0.25 + 
                           edge_score * 0.15)
            
            assessment['scores']['overall_quality'] = float(overall_score)
        else:
            assessment['scores']['overall_quality'] = 0.0
        
        return assessment
    
    def select_quality_frames(self, frame_indices, max_attempts=None, strict_mode=False):
        """
        Select high-quality frames from a list of candidate frame indices
        
        Args:
            frame_indices: List of frame indices to evaluate
            max_attempts: Maximum number of frames to check (None = check all)
            strict_mode: Apply stricter quality criteria
            
        Returns:
            List of (frame_index, frame, quality_score) tuples for suitable frames
        """
        if max_attempts is None:
            max_attempts = len(frame_indices)
        
        quality_frames = []
        frames_checked = 0
        previous_frame = None
        
        logger.debug(f"Evaluating frame quality (strict_mode={strict_mode})...")
        
        for idx in frame_indices[:max_attempts]:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            
            if not ret:
                continue
                
            frames_checked += 1
            
            # Assess frame quality
            assessment = self.assess_frame_quality(frame, previous_frame, strict_mode)
            
            if assessment['is_suitable']:
                quality_frames.append((idx, frame, assessment['scores']['overall_quality']))
            else:
                if frames_checked <= 10:  # Only log first few rejections to avoid spam
                    logger.debug(f"  Frame {idx} ({idx/self.fps:.1f}s) rejected: {', '.join(assessment['reasons_rejected'])}")
            
            previous_frame = frame
        
        # Sort by quality score (highest first)
        quality_frames.sort(key=lambda x: x[2], reverse=True)

        # Store for potential reuse
        self._quality_frames_cache = quality_frames
        
        logger.debug(f"  Found {len(quality_frames)} suitable frames out of {frames_checked} checked")
        
        return quality_frames
    
    def detect_blanking_borders(self, threshold=10, edge_sample_width=100):
        """
        Detect borders with enhanced frame quality filtering
        """
        frame_indices = np.linspace(0, self.total_frames - 1, 
                                   min(self.sample_frames, self.total_frames), 
                                   dtype=int)
        
        # Select only high-quality frames for border detection
        quality_frames = self.select_quality_frames(frame_indices, strict_mode=False)
        
        if len(quality_frames) < 5:
            logger.debug(f"⚠️ Only {len(quality_frames)} suitable frames found, relaxing criteria...")
            # Try again with less strict criteria
            quality_frames = self.select_quality_frames(frame_indices, strict_mode=False)
            
        if len(quality_frames) == 0:
            logger.debug("⚠️ No suitable frames found for border detection")
            return None
        
        logger.debug(f"Using {len(quality_frames)} high-quality frames for border detection...")

        # Check for vertical blanking lines
        vertical_blanking = self.detect_vertical_blanking_lines(quality_frames)
        
        left_borders = []
        right_borders = []
        top_borders = []
        bottom_borders = []
        
        for frame_idx, frame, quality_score in quality_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Detect left border - scan from left
            left = 0
            for x in range(min(edge_sample_width, w)):
                if np.mean(gray[:, x]) > threshold:
                    left = x
                    break
                    
            # Detect right border - scan from right
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
            
        # Calculate stable borders
        median_left = int(np.median(left_borders))
        median_right = int(np.median(right_borders))

        if vertical_blanking.get('left_blanking'):
            detected_left = vertical_blanking['left_blanking'] + 2  # Add small buffer
            if detected_left > median_left:
                logger.debug(f"  Vertical blanking line detected at x={vertical_blanking['left_blanking']}")
                logger.debug(f"  Adjusting left border from {median_left} to {detected_left}")
                median_left = detected_left
        
        if vertical_blanking.get('right_blanking'):
            detected_right = vertical_blanking['right_blanking'] - 2  # Subtract small buffer
            if detected_right < median_right:
                logger.debug(f"  Vertical blanking line detected at x={vertical_blanking['right_blanking']}")
                logger.debug(f"  Adjusting right border from {median_right} to {detected_right}")
                median_right = detected_right
        
        # Add padding for tighter active area
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
            logger.debug("Warning: Detected active area seems too small")
            return None
            
        result = (median_left, mode_top, active_width, active_height)
        
        logger.debug(f"\nBorder detection statistics:")
        logger.debug(f"  Left border: median={median_left} (std={np.std(left_borders):.1f})")
        logger.debug(f"  Right border: median={median_right} (std={np.std(right_borders):.1f})")
        logger.debug(f"  Top border: mode={mode_top} (variations={len(top_unique)})")
        logger.debug(f"  Bottom border: mode={mode_bottom} (variations={len(bottom_unique)})")
        logger.debug(f"  Active area (with {padding}px padding): {active_width}x{active_height} at ({median_left},{mode_top})")
        
        return result
    
    def detect_vertical_blanking_lines(self, frames, edge_width=30):
        """
        Detect vertical blanking lines that might be missed by standard border detection.
        These often appear as consistent vertical lines of violations in analog video.
        
        Args:
            frames: List of (frame_idx, frame, quality_score) tuples
            edge_width: How far from the edge to scan for vertical lines
            
        Returns:
            dict: Detected blanking line positions for each edge
        """
        blanking_lines = {
            'left': [],
            'right': []
        }
        
        for frame_idx, frame, _ in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Check left edge for vertical blanking lines
            left_region = gray[:, :edge_width]
            for x in range(edge_width):
                column = left_region[:, x]
                # Check if this column is consistently dark (potential blanking)
                if np.mean(column) < 20:  # Dark threshold
                    # Check for consistency (low variation)
                    if np.std(column) < 10:
                        blanking_lines['left'].append(x)
                        
            # Check right edge
            right_region = gray[:, -edge_width:]
            for x in range(edge_width):
                column = right_region[:, x]
                if np.mean(column) < 20 and np.std(column) < 10:
                    blanking_lines['right'].append(w - edge_width + x)
        
        # Find the most common blanking positions
        result = {}
        if blanking_lines['left']:
            # Find the rightmost consistent blanking line on the left
            left_counts = np.bincount(blanking_lines['left'])
            consistent_lines = np.where(left_counts >= len(frames) * 0.3)[0]  # 30% threshold
            if len(consistent_lines) > 0:
                result['left_blanking'] = int(np.max(consistent_lines))
        
        if blanking_lines['right']:
            # Find the leftmost consistent blanking line on the right
            right_counts = np.bincount([w - 1 - x for x in blanking_lines['right']])
            consistent_lines = np.where(right_counts >= len(frames) * 0.3)[0]
            if len(consistent_lines) > 0:
                result['right_blanking'] = int(w - 1 - np.max(consistent_lines))
        
        return result

    
    def detect_head_switching_artifacts(self, active_area=None, sample_frames=20):
        """
        Detect head switching artifacts with enhanced frame quality filtering
        """
        try:
            if active_area:
                crop_x, crop_y, crop_w, crop_h = active_area
                analysis_y = crop_y + crop_h - 10
                analysis_height = 10
            else:
                crop_x, crop_y, crop_w, crop_h = 0, 0, self.width, self.height
                analysis_y = self.height - 15
                analysis_height = 15
            
            analysis_y = max(0, analysis_y)
            analysis_height = min(analysis_height, self.height - analysis_y)
            
            if analysis_height <= 0 or crop_w <= 0:
                logger.debug("⚠️ Invalid analysis region for head switching detection")
                return {
                    'frames_analyzed': 0,
                    'frames_with_artifacts': 0,
                    'artifact_percentage': 0.0,
                    'severity': 'none',
                    'error': 'Invalid analysis region'
                }
            
            logger.debug(f"\nAnalyzing bottom {analysis_height} lines for head switching artifacts...")
            logger.debug(f"Analysis region: {crop_w}x{analysis_height} at ({crop_x},{analysis_y})")
            
            # Sample frames for analysis
            frame_indices = np.linspace(0, self.total_frames - 1, 
                                       min(sample_frames, self.total_frames), 
                                       dtype=int)
            
            # Select only quality frames for artifact detection
            quality_frames = self.select_quality_frames(frame_indices, strict_mode=False)
            
            if len(quality_frames) < 3:
                logger.debug(f"⚠️ Only {len(quality_frames)} suitable frames for head switching analysis")
            
            artifact_detections = []
            line_asymmetry_scores = []
            horizontal_discontinuities = []
            
            for frame_idx, frame, quality_score in quality_frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Extract the bottom region for analysis
                if analysis_y + analysis_height > gray.shape[0] or crop_x + crop_w > gray.shape[1]:
                    continue
                    
                bottom_region = gray[analysis_y:analysis_y + analysis_height, crop_x:crop_x + crop_w]
                
                if bottom_region.size == 0:
                    continue
                
                # Analyze each line in the bottom region
                frame_asymmetries = []
                frame_discontinuities = []
                
                for line_idx in range(bottom_region.shape[0]):
                    line = bottom_region[line_idx, :]
                    line_width = len(line)
                    
                    if line_width < 20:
                        continue
                    
                    # Split line into left and right halves
                    mid_point = line_width // 2
                    left_half = line[:mid_point]
                    right_half = line[mid_point:]
                    
                    # Calculate brightness for each half
                    left_brightness = np.mean(left_half)
                    right_brightness = np.mean(right_half)
                    
                    # Calculate asymmetry score
                    if left_brightness > 10:
                        asymmetry = abs(left_brightness - right_brightness) / left_brightness
                    else:
                        asymmetry = 0
                    
                    frame_asymmetries.append(asymmetry)
                    
                    # Look for horizontal discontinuities and track their positions
                    discontinuity_info = None
                    if left_brightness > 30 and right_brightness < 15:
                        discontinuity_info = {
                            'line': line_idx,
                            'start_x': mid_point,  # Discontinuity starts at midpoint
                            'end_x': line_width - 1,
                            'type': 'left_right_split'
                        }
                        frame_discontinuities.append(discontinuity_info)
                    
                    # Check for sharp transitions within the line
                    if line_width > 10:
                        right_quarter = line[int(line_width * 0.75):]
                        left_three_quarters = line[:int(line_width * 0.75)]
                        
                        if len(right_quarter) > 0 and len(left_three_quarters) > 0:
                            if np.mean(left_three_quarters) > 30 and np.mean(right_quarter) < 15:
                                # This discontinuity affects the right quarter
                                discontinuity_info = {
                                    'line': line_idx,
                                    'start_x': int(line_width * 0.75),
                                    'end_x': line_width - 1,
                                    'type': 'right_quarter_drop'
                                }
                                frame_discontinuities.append(discontinuity_info)
                
                # Store results for this frame
                if frame_asymmetries:
                    avg_asymmetry = np.mean(frame_asymmetries)
                    max_asymmetry = np.max(frame_asymmetries)
                    line_asymmetry_scores.append(avg_asymmetry)
                    
                    # Consider it an artifact if multiple lines show high asymmetry
                    artifact_lines = sum(1 for asym in frame_asymmetries if asym > 0.5)
                    if artifact_lines >= 2:
                        # Calculate the affected region based on discontinuities
                        if frame_discontinuities:
                            # Find the earliest start and latest end of discontinuities
                            start_positions = [d['start_x'] for d in frame_discontinuities if isinstance(d, dict)]
                            end_positions = [d['end_x'] for d in frame_discontinuities if isinstance(d, dict)]
                            
                            if start_positions and end_positions:
                                artifact_start_x = min(start_positions)
                                artifact_end_x = max(end_positions)
                            else:
                                # Fallback to right half if no specific positions
                                artifact_start_x = line_width // 2
                                artifact_end_x = line_width - 1
                        else:
                            # Default to right half for asymmetry-based artifacts
                            artifact_start_x = line_width // 2
                            artifact_end_x = line_width - 1
                        
                        artifact_detections.append({
                            'frame_idx': int(frame_idx),
                            'time': float(frame_idx / self.fps),
                            'avg_asymmetry': float(avg_asymmetry),
                            'max_asymmetry': float(max_asymmetry),
                            'artifact_lines': int(artifact_lines),
                            'discontinuities': len([d for d in frame_discontinuities if isinstance(d, dict)]),
                            'quality_score': float(quality_score),
                            'artifact_region': {
                                'start_x': int(artifact_start_x),
                                'end_x': int(artifact_end_x),
                                'width': int(artifact_end_x - artifact_start_x + 1)
                            }
                        })
                
                horizontal_discontinuities.extend([d for d in frame_discontinuities if isinstance(d, dict)])
            
            # Analyze results
            total_quality_frames = len(quality_frames)
            
            # Calculate typical artifact region if artifacts were detected
            artifact_region_info = None
            if artifact_detections:
                start_positions = []
                end_positions = []
                widths = []
                
                for detection in artifact_detections:
                    if 'artifact_region' in detection:
                        start_positions.append(detection['artifact_region']['start_x'])
                        end_positions.append(detection['artifact_region']['end_x'])
                        widths.append(detection['artifact_region']['width'])
                
                if start_positions and end_positions:
                    # Use median values for a typical artifact region
                    typical_start = int(np.median(start_positions))
                    typical_end = int(np.median(end_positions))
                    typical_width = int(np.median(widths))
                    
                    artifact_region_info = {
                        'start_x': typical_start,
                        'end_x': typical_end,
                        'width': typical_width,
                        'relative_start': typical_start / crop_w if crop_w > 0 else 0.5,
                        'relative_end': typical_end / crop_w if crop_w > 0 else 1.0
                    }
            
            results = {
                'frames_analyzed': total_quality_frames,
                'frames_with_artifacts': len(artifact_detections),
                'artifact_percentage': 0.0,
                'avg_asymmetry': 0.0,
                'max_asymmetry': 0.0,
                'total_discontinuities': len(horizontal_discontinuities),
                'analysis_region': {
                    'x': int(crop_x),
                    'y': int(analysis_y), 
                    'width': int(crop_w),
                    'height': int(analysis_height)
                },
                'artifact_frames': artifact_detections[:5],
                'severity': 'none',
                'quality_frames_used': total_quality_frames,
                'total_frames_sampled': len(frame_indices),
                'artifact_region': artifact_region_info
            }
            
            if line_asymmetry_scores:
                results['avg_asymmetry'] = float(np.mean(line_asymmetry_scores))
                results['max_asymmetry'] = float(np.max(line_asymmetry_scores))
            
            if total_quality_frames > 0:
                results['artifact_percentage'] = float((len(artifact_detections) / total_quality_frames) * 100)
            
            # Determine severity
            if results['artifact_percentage'] > 50:
                results['severity'] = 'severe'
            elif results['artifact_percentage'] > 20:
                results['severity'] = 'moderate'  
            elif results['artifact_percentage'] > 5:
                results['severity'] = 'minor'
            
            # Report findings
            logger.debug(f"  Quality frames used: {total_quality_frames}/{len(frame_indices)}")
            logger.debug(f"  Frames with head switching artifacts: {results['frames_with_artifacts']}/{results['frames_analyzed']} ({results['artifact_percentage']:.1f}%)")
            logger.debug(f"  Average asymmetry score: {results['avg_asymmetry']:.3f}")
            logger.debug(f"  Horizontal discontinuities found: {results['total_discontinuities']}")
            logger.debug(f"  Severity: {results['severity']}")
            
            if results['severity'] != 'none':
                logger.debug(f"  ⚠️  Head switching artifacts detected - check bottom of picture area")
                if artifact_region_info:
                    logger.debug(f"  Artifact region: {artifact_region_info['width']}px wide, starting at {artifact_region_info['relative_start']:.1%} of scan line")
            else:
                logger.info(f"  ✓ No significant head switching artifacts detected")
            
            return results
            
        except Exception as e:
            logger.debug(f"⚠️ Error in head switching analysis: {e}")
            return {
                'frames_analyzed': 0,
                'frames_with_artifacts': 0,
                'artifact_percentage': 0.0,
                'severity': 'error',
                'error': str(e)
            }
    
    def analyze_border_regions(self, active_area):
        """
        Get information about border regions
        Returns dictionary with border region coordinates
        """
        if not active_area:
            logger.debug("No active area detected, cannot analyze borders")
            return None
            
        x, y, w, h = active_area
        regions = {}
        
        # Left border
        if x > 10:
            regions['left_border'] = (0, 0, int(x), int(self.height))
            logger.debug(f"Left border region: {x}px wide")
        
        # Right border  
        right_border_start = x + w
        if right_border_start < self.width - 10:
            right_width = self.width - right_border_start
            regions['right_border'] = (int(right_border_start), 0, int(right_width), int(self.height))
            logger.debug(f"Right border region: {right_width}px wide\n")
        
        # Top border
        if y > 10:
            regions['top_border'] = (0, 0, int(self.width), int(y))
            logger.debug(f"Top border region: {y}px tall")
        
        # Bottom border
        bottom_border_start = y + h
        if bottom_border_start < self.height - 10:
            bottom_height = self.height - bottom_border_start
            regions['bottom_border'] = (0, int(bottom_border_start), int(self.width), int(bottom_height))
            logger.debug(f"Bottom border region: {bottom_height}px tall")
            
        return regions
    
    def find_good_representative_frame(self, target_time=150, search_window=120):
        """
        Find a good representative frame using enhanced quality assessment
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
        
        # Check frames every 1 second in the search window
        check_interval = max(1, int(self.fps))
        frame_indices = list(range(start_frame, end_frame, check_interval))
        
        # Use quality assessment to find the best frame
        quality_frames = self.select_quality_frames(frame_indices, strict_mode=True)
        
        if quality_frames:
            # Return the highest quality frame
            best_frame_idx, best_frame, best_score = quality_frames[0]
            logger.info(f"✓ Selected high-quality frame at {best_frame_idx/self.fps:.1f}s (quality score: {best_score:.3f})")
            return best_frame
        else:
            # Fallback: try with less strict criteria
            logger.debug("No frames met strict criteria, trying with relaxed standards...")
            quality_frames = self.select_quality_frames(frame_indices, strict_mode=False)
            
            if quality_frames:
                best_frame_idx, best_frame, best_score = quality_frames[0]
                logger.info(f"✓ Selected frame at {best_frame_idx/self.fps:.1f}s (relaxed quality score: {best_score:.3f})")
                return best_frame
            else:
                # Final fallback
                logger.warning(f"⚠️ No suitable frame found, using target frame as fallback")
                fallback_frame = target_frame if target_frame < self.total_frames else self.total_frames // 2
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_frame)
                ret, frame = self.cap.read()
                return frame if ret else None

    def generate_border_visualization(self, output_path, active_area=None, head_switching_results=None, target_time=150, search_window=120):
        """
        Generate visual showing detected borders and active area
        """
        frame = self.find_good_representative_frame(target_time, search_window)
        
        if frame is None:
            return False
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Full frame with regions marked
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax1.imshow(frame_rgb)
        ax1.set_title('Full Frame with Border Detection')
        ax1.axis('off')
        
        if active_area:
            x, y, w, h = active_area
            
            # Mark border regions in red
            border_added = False
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
            
            # Highlight head switching analysis region if artifacts detected
            if head_switching_results and head_switching_results.get('severity') != 'none':
                hs_region = head_switching_results.get('analysis_region', {})
                artifact_region = head_switching_results.get('artifact_region', {})
                
                if hs_region and artifact_region:
                    hs_x = hs_region.get('x', 0)
                    hs_y = hs_region.get('y', self.height - 10)
                    hs_w = hs_region.get('width', self.width)
                    
                    # Calculate the actual pixel positions of the artifact region
                    artifact_start_x = hs_x + artifact_region.get('start_x', hs_w // 2)
                    artifact_end_x = hs_x + artifact_region.get('end_x', hs_w - 1)
                    
                    # Ensure the line doesn't go beyond the analysis region
                    artifact_start_x = max(hs_x, min(artifact_start_x, hs_x + hs_w))
                    artifact_end_x = max(hs_x, min(artifact_end_x, hs_x + hs_w))
                    
                    line_y = hs_y + hs_region.get('height', 10) - 1
                    ax1.plot([artifact_start_x, artifact_end_x], [line_y, line_y], 
                            color='orange', linewidth=3, alpha=0.9, 
                            label='Head Switching Artifacts')
                elif hs_region:
                    # Fallback to original behavior if no specific artifact region
                    hs_x = hs_region.get('x', 0)
                    hs_y = hs_region.get('y', self.height - 10)
                    hs_w = hs_region.get('width', self.width)
                    
                    line_y = hs_y + hs_region.get('height', 10) - 1
                    ax1.plot([hs_x, hs_x + hs_w], [line_y, line_y], 
                            color='orange', linewidth=2, alpha=0.8, 
                            label='Head Switching Artifacts')
            
            if border_added or (head_switching_results and head_switching_results.get('severity') != 'none'):
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
            if head_switching_results and head_switching_results.get('severity') != 'none':
                hs_info = f"Head switching artifacts: {head_switching_results['severity']} ({head_switching_results['artifact_percentage']:.1f}% of frames)"
                fig.text(0.5, 0.95, hs_info, ha='center', fontsize=12, weight='bold', color='orange')
            else:
                fig.text(0.5, 0.95, 'No head switching artifacts detected', ha='center', fontsize=12, weight='bold', color='green')
        else:
            ax2.text(0.5, 0.5, 'No borders detected', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
            
            # Still show head switching info even without borders
            if head_switching_results and head_switching_results.get('severity') != 'none':
                hs_region = head_switching_results.get('analysis_region', {})
                artifact_region = head_switching_results.get('artifact_region', {})
                
                if hs_region and artifact_region:
                    hs_x = hs_region.get('x', 0)
                    hs_y = hs_region.get('y', self.height - 15)
                    hs_w = hs_region.get('width', self.width)
                    
                    # Calculate the actual pixel positions of the artifact region
                    artifact_start_x = hs_x + artifact_region.get('start_x', hs_w // 2)
                    artifact_end_x = hs_x + artifact_region.get('end_x', hs_w - 1)
                    
                    # Ensure the line doesn't go beyond the analysis region
                    artifact_start_x = max(hs_x, min(artifact_start_x, hs_x + hs_w))
                    artifact_end_x = max(hs_x, min(artifact_end_x, hs_x + hs_w))
                    
                    line_y = hs_y + hs_region.get('height', 15) - 1
                    ax1.plot([artifact_start_x, artifact_end_x], [line_y, line_y], 
                            color='orange', linewidth=3, alpha=0.9, 
                            label='Head Switching Artifacts')
                    ax1.legend()
                elif hs_region:
                    # Fallback to original behavior if no specific artifact region
                    hs_x = hs_region.get('x', 0)
                    hs_y = hs_region.get('y', self.height - 15)
                    hs_w = hs_region.get('width', self.width)
                    
                    line_y = hs_y + hs_region.get('height', 15) - 1
                    ax1.plot([hs_x, hs_x + hs_w], [line_y, line_y], 
                            color='orange', linewidth=2, alpha=0.8, 
                            label='Head Switching Artifacts')
                    ax1.legend()
                
                hs_info = f"Head switching artifacts: {head_switching_results['severity']} ({head_switching_results['artifact_percentage']:.1f}% of frames)"
                fig.text(0.5, 0.95, hs_info, ha='center', fontsize=12, weight='bold', color='orange')
            else:
                fig.text(0.5, 0.95, 'No head switching artifacts detected', ha='center', fontsize=12, weight='bold', color='green')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    
    def save_border_data(self, output_path, active_area=None, head_switching_results=None):
        """
        Save border detection results to JSON file with quality frame hints
        """
        border_regions = None
        if active_area:
            border_regions = self.analyze_border_regions(active_area)
        
        # Collect quality frame hints if using sophisticated detection
        quality_frame_hints = []
        if hasattr(self, '_quality_frames_cache'):
            # Store the best quality frames found during detection
            for frame_idx, frame, quality_score in self._quality_frames_cache[:10]:
                time_seconds = frame_idx / self.fps
                quality_frame_hints.append((time_seconds, quality_score))
        
        data = {
            'video_file': self.video_path,
            'video_properties': {
                'width': int(self.width),
                'height': int(self.height),
                'fps': float(self.fps),
                'duration': float(self.duration),
                'total_frames': int(self.total_frames)
            },
            'active_area': [int(x) for x in active_area] if active_area else None,
            'border_regions': border_regions,
            'head_switching_artifacts': head_switching_results,
            'quality_frame_hints': quality_frame_hints,  # Add this
            'detection_settings': {
                'sample_frames': int(self.sample_frames),
                'threshold': 10,
                'padding': 5,
                'quality_filtering': True
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data
    
    def close(self):
        """Close video capture"""
        self.cap.release()


def detect_video_borders(video_path, output_dir=None, target_viz_time=150, search_window=120,
                        threshold=10, edge_sample_width=100, sample_frames=30, padding=5):
    """
    Main function to detect borders in a video file with enhanced quality filtering
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for results  
        target_viz_time: Target time for visualization frame
        search_window: Window to search for good frame
        threshold: Brightness threshold for border detection (default: 10)
        edge_sample_width: How far from edges to scan (default: 100)
        sample_frames: Number of frames to sample (default: 30)
        padding: Extra pixels to add for tighter active area (default: 5)
    """
    video_path = Path(video_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = video_path.parent
        
    logger.debug(f"Processing: {video_path.name}")
    logger.debug("Detecting blanking borders and active picture area with quality filtering...")
    
    detector = VideoBorderDetector(video_path)
    
    # Detect borders with quality filtering
    active_area = detector.detect_blanking_borders(threshold=10)
    
    # Detect head switching artifacts with quality filtering
    head_switching_results = detector.detect_head_switching_artifacts(active_area)
    
    if active_area:
        logger.info(f"\n✓ Active area detected: {active_area[2]}x{active_area[3]} at ({active_area[0]},{active_area[1]})")
        
        # Generate visualization
        viz_path = output_dir / f"{video_path.stem}_border_detection.jpg"
        detector.generate_border_visualization(viz_path, active_area, head_switching_results, target_viz_time, search_window)
        logger.info(f"✓ Visualization saved: {viz_path}\n")
        
        # Save border data
        data_path = output_dir / f"{video_path.stem}_border_data.json"
        results = detector.save_border_data(data_path, active_area, head_switching_results)
        logger.info(f"✓ Border data saved: {data_path}\n")
        
    else:
        logger.warning("⚠️ No clear borders detected")
        
        # Still generate visualization to show head switching artifacts
        viz_path = output_dir / f"{video_path.stem}_border_detection.jpg"
        detector.generate_border_visualization(viz_path, None, head_switching_results, target_viz_time, search_window)
        logger.info(f"✓ Visualization saved: {viz_path}")
        
        results = detector.save_border_data(
            output_dir / f"{video_path.stem}_border_data.json",
            active_area, 
            head_switching_results
        )
        logger.info(f"✓ Border data saved: {output_dir / f'{video_path.stem}_border_data.json'}")

    # Store quality frames for signalstats to use
    quality_frame_hints = []
    if hasattr(detector, '_quality_frames_cache'):
        for frame_idx, frame, quality_score in detector._quality_frames_cache[:10]:
            time_seconds = frame_idx / detector.fps
            quality_frame_hints.append([time_seconds, quality_score])
        results['quality_frame_hints'] = quality_frame_hints
    
    detector.close()
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        viz_time = int(sys.argv[2]) if len(sys.argv) > 2 else 150
        search_window = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    else:
        video_file = "JPC_AV_00011.mkv"
        viz_time = 150
        search_window = 120
    
    logger.debug(f"Using target time: {viz_time}s, search window: {search_window}s")
    results = detect_video_borders(video_file, target_viz_time=viz_time, search_window=search_window)
    
    if results['active_area']:
        x, y, w, h = results['active_area']
        logger.info(f"\nBorder Detection Summary:")
        logger.info(f"Active area: {w}x{h} at position ({x},{y})")
        logger.info(f"Borders: L={x}px, R={results['video_properties']['width']-x-w}px, T={y}px, B={results['video_properties']['height']-y-h}px")
    else:
        logger.info("\nNo borders detected - video appears to be full frame active content")
    
    # Head switching artifact summary
    if results.get('head_switching_artifacts'):
        hs_results = results['head_switching_artifacts']
        logger.info(f"\nHead Switching Artifact Analysis (Quality Filtering Applied):")
        logger.debug(f"Quality frames used: {hs_results.get('quality_frames_used', 'N/A')}/{hs_results.get('total_frames_sampled', 'N/A')}")
        logger.debug(f"Severity: {hs_results['severity']}")
        if hs_results['severity'] != 'none':
            logger.debug(f"Affected frames: {hs_results['frames_with_artifacts']}/{hs_results['frames_analyzed']} ({hs_results['artifact_percentage']:.1f}%)")
        else:
            logger.info(f"✓ No significant head switching artifacts detected")