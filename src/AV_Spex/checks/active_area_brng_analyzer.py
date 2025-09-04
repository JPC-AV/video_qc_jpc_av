#!/usr/bin/env python3
"""
Active Area BRNG Analyzer

Key improvements:
- Skips color bars and test patterns at video start
- Better boundary artifact detection for triggering border re-analysis
- More robust edge detection to identify missed borders
- Clearer reporting of issues requiring action
"""

import subprocess
import cv2
import numpy as np
import json
from pathlib import Path
import sys
from collections import defaultdict
from scipy import ndimage, signal
import shlex

from AV_Spex.utils.log_setup import logger


class ActiveAreaBrngAnalyzer:
    """
    Analyzes BRNG violations in the active picture area with enhanced pattern recognition
    """
    
    def __init__(self, video_path, border_data_path=None, output_dir=None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent
        self.output_dir.mkdir(exist_ok=True)

        # Initialize skip offset
        self.skip_offset = 0
        
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
        self.original_video = self.temp_dir / f"{self.video_path.stem}_original.mp4"
        self.analysis_output = self.output_dir / f"{self.video_path.stem}_active_brng_analysis.json"
        
        # Get video properties
        self._get_video_properties()
        
    def _get_video_properties(self):
        """Get basic video properties"""
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def load_border_data(self, border_data_path):
        """Load border detection data"""
        try:
            with open(border_data_path, 'r') as f:
                self.border_data = json.load(f)
            
            if self.border_data and self.border_data.get('active_area'):
                self.active_area = tuple(self.border_data['active_area'])
                logger.debug(f"✔ Loaded border data. Active area: {self.active_area}\n")
            else:
                logger.debug("⚠️ Border data doesn't contain active area")
        except Exception as e:
            logger.debug(f"⚠️ Could not load border data: {e}")
    
    def detect_color_bars_or_test_pattern(self, frame):
        """
        Detect if frame contains color bars or test patterns
        Returns True if frame appears to be a test pattern
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check for vertical bars (common in SMPTE color bars)
        # Sample vertical stripes
        stripe_width = w // 8
        stripe_colors = []
        
        for i in range(8):
            x_start = i * stripe_width
            x_end = min((i + 1) * stripe_width, w)
            stripe_region = frame[:h//2, x_start:x_end]
            
            if stripe_region.size > 0:
                mean_color = np.mean(stripe_region, axis=(0, 1))
                stripe_colors.append(mean_color)
        
        if len(stripe_colors) >= 7:
            # Check if colors match typical color bar pattern
            # High saturation, distinct colors
            color_diffs = []
            for i in range(len(stripe_colors) - 1):
                diff = np.linalg.norm(stripe_colors[i] - stripe_colors[i+1])
                color_diffs.append(diff)
            
            # Color bars have high differences between adjacent stripes
            if np.mean(color_diffs) > 50:
                # Additional check: bottom portion often has specific patterns
                bottom_region = gray[int(h*0.75):, :]
                bottom_std = np.std(bottom_region)
                
                # Color bars often have low variation in bottom sections
                if bottom_std < 40 or bottom_std > 80:
                    return True
        
        # Check for circle pattern (common in test patterns)
        # Look for circular edges in center
        center_region = gray[h//3:2*h//3, w//3:2*w//3]
        circles = cv2.HoughCircles(
            center_region,
            cv2.HOUGH_GRADIENT,
            1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=min(h, w)//4
        )
        
        if circles is not None and len(circles[0]) > 0:
            return True
        
        # Check for grid patterns (another common test pattern)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is not None and len(lines) > 20:
            # Many straight lines might indicate a grid pattern
            horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 5)
            vertical_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 5)
            
            if horizontal_lines > 5 and vertical_lines > 5:
                return True
        
        return False
    
    def find_content_start(self, cap, max_seconds=30):
        """
        Find where actual content starts (skip color bars, black frames, etc.)
        Returns frame index where content begins
        """
        logger.debug(f"  Detecting content start (skipping test patterns)...")
        
        check_interval = max(1, int(self.fps))  # Check every second
        max_frames = int(max_seconds * self.fps)
        
        for frame_idx in range(0, min(max_frames, self.total_frames), check_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Check if it's a test pattern
            if self.detect_color_bars_or_test_pattern(frame):
                continue
            
            # Check if it's a black frame
            if np.mean(frame) < 10:
                continue
            
            # Check if it has reasonable content characteristics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.std(gray) > 15:  # Has some variation
                logger.debug(f"  Content starts at frame {frame_idx} ({frame_idx/self.fps:.1f}s)")
                return frame_idx
        
        logger.debug(f"  No clear content start found, using frame 0")
        return 0
    
    def process_with_ffmpeg(self, duration_limit=300, skip_start_seconds=0):
        """
        Process video with ffmpeg signalstats, creating both highlighted and original versions
        for differential analysis.
        """
        logger.debug(f"\nProcessing video with ffmpeg signalstats...")
        
        # Build filter chain for active area crop
        crop_filter = ""
        if self.active_area:
            x, y, w, h = self.active_area
            crop_filter = f"crop={w}:{h}:{x}:{y},"
            logger.debug(f"  Cropping to active area: {w}x{h} at ({x},{y})")
        else:
            logger.debug("  No border data - analyzing full frame")
        
        # Add seek position if skipping start
        seek_args = []
        if skip_start_seconds > 0:
            seek_args = ["-ss", str(skip_start_seconds)]
            logger.debug(f"  Skipping first {skip_start_seconds} seconds")
        
        # Create highlighted version with BRNG violations marked in cyan
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
            str(self.highlighted_video)
        ]
        
        # Create original version with same crop for comparison
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
            str(self.original_video)
        ]
        
        logger.debug(f"  Processing up to {duration_limit} seconds...")
        
        try:
            # Process highlighted version
            logger.debug("  Creating highlighted version...")
            result = subprocess.run(highlighted_cmd, capture_output=True, text=True, check=True)
            
            # Process original version
            logger.debug("  Creating original version for comparison...")
            result = subprocess.run(original_cmd, capture_output=True, text=True, check=True)
            
            logger.debug(f"✔ FFmpeg processing complete")
            return True
        except subprocess.CalledProcessError as e:
            logger.debug(f"✗ FFmpeg error: {e}")
            if e.stderr:
                logger.warning(f"  Error details: {e.stderr[:500]}")
            return False
    
    def detect_brng_violations_differential(self, frame_highlighted, frame_original):
        """
        Detect BRNG violations by comparing highlighted frame with original.
        This eliminates false positives from naturally cyan content.
        """
        # Calculate the difference between highlighted and original
        diff = cv2.absdiff(frame_highlighted, frame_original)
        
        # Convert to HSV to isolate cyan changes
        diff_hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
        
        # Look for significant changes in the cyan range
        cyan_changes = cv2.inRange(diff_hsv, 
                                  np.array([80, 30, 30]),   # Lower threshold
                                  np.array([110, 255, 255])) # Upper threshold
        
        # Additional filtering: remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cyan_changes = cv2.morphologyEx(cyan_changes, cv2.MORPH_OPEN, kernel)
        
        return cyan_changes
    
    def detect_edge_violations(self, violation_mask, edge_width=10):
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
            'blanking_depth': {},  # How far the blanking extends from edge
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
                # For left/right edges, check for horizontal lines of violations
                for row in range(edge_region.shape[0]):
                    row_violations = edge_region[row, :]
                    if np.any(row_violations > 0):
                        # Check if violations form a line pattern (allowing gaps)
                        violation_positions = np.where(row_violations > 0)[0]
                        if len(violation_positions) >= 2:
                            # Check if violations are roughly aligned (within 3 pixels of edge)
                            if edge_name == 'left' and np.max(violation_positions) <= 3:
                                linear_score += 1
                            elif edge_name == 'right' and np.min(violation_positions) >= edge_width - 4:
                                linear_score += 1
                
                # Calculate linear pattern percentage
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
            if violation_percentage > 5:
                if orientation == 'vertical':
                    # Find the deepest violation from the edge
                    max_depth = 0
                    for row in range(edge_region.shape[0]):
                        row_violations = np.where(edge_region[row, :] > 0)[0]
                        if len(row_violations) > 0:
                            if edge_name == 'left':
                                depth = np.max(row_violations)
                            else:  # right
                                depth = edge_width - np.min(row_violations)
                            max_depth = max(max_depth, depth)
                else:  # horizontal
                    max_depth = 0
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
            # Lower thresholds for better detection
            if violation_percentage > 5 or linear_percentage > 30:
                edge_info['edges_affected'].append(edge_name)
                edge_info['has_edge_violations'] = True
                
                # Mark as continuous if we have strong linear patterns
                if linear_percentage > 50:
                    edge_info['continuous_edges'].append(edge_name)
                
                # Calculate recommended expansion based on blanking depth
                if edge_name in edge_info['blanking_depth']:
                    # Recommend expanding by at least the blanking depth plus buffer
                    recommended_expansion = edge_info['blanking_depth'][edge_name] + 5
                    edge_info['expansion_recommendations'][edge_name] = recommended_expansion
        
        # Refine severity assessment
        if len(edge_info['continuous_edges']) >= 2:
            edge_info['severity'] = 'high'
        elif len(edge_info['continuous_edges']) >= 1:
            edge_info['severity'] = 'medium'
        elif len(edge_info['edges_affected']) >= 2:
            edge_info['severity'] = 'low'
        elif len(edge_info['edges_affected']) >= 1 and max(edge_info['linear_patterns'].values(), default=0) > 30:
            edge_info['severity'] = 'low'
        
        return edge_info
    
    def analyze_violation_patterns(self, violation_mask, frame):
        """
        Analyze BRNG violations with enhanced edge pattern detection
        """
        h, w = violation_mask.shape
        
        # Use improved edge detection
        edge_violations = self.detect_edge_violations(violation_mask, edge_width=15)
        
        # Edge-based analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check if violations occur at edges
        edge_violations_mask = cv2.bitwise_and(violation_mask, edges)
        edge_violation_ratio = np.sum(edge_violations_mask > 0) / max(1, np.sum(violation_mask > 0))
        
        # Spatial patterns with linear pattern info
        spatial_patterns = {
            'edge_concentrated': bool(edge_violation_ratio > 0.6),
            'has_boundary_artifacts': edge_violations['has_edge_violations'],
            'boundary_edges': edge_violations['edges_affected'],
            'boundary_severity': edge_violations['severity'],
            'linear_patterns_detected': any(v > 30 for v in edge_violations['linear_patterns'].values())
        }
        
        # Analyze in context of luma zones (but not if strong edge violations)
        luma_distribution = {}
        if not (edge_violations['has_edge_violations'] and edge_violations['severity'] in ['medium', 'high']):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luma_distribution = self._analyze_luma_zone_violations(violation_mask, gray)
        
        # Generate diagnostic with enhanced information
        diagnostic = self._generate_enhanced_diagnostic(spatial_patterns, edge_violations, luma_distribution)
        
        return {
            'spatial_patterns': spatial_patterns,
            'edge_violations': edge_violations,
            'luma_distribution': luma_distribution,
            'edge_violation_ratio': float(edge_violation_ratio),
            'diagnostic': diagnostic,
            'boundary_artifacts': edge_violations,
            'linear_pattern_info': edge_violations.get('linear_patterns', {}),
            'expansion_recommendations': edge_violations.get('expansion_recommendations', {})
        }


    def _generate_enhanced_diagnostic(self, spatial_patterns, edge_violations, luma_distribution):
        """Generate enhanced diagnostic with linear pattern information"""
        diagnostics = []
        
        # Check for boundary artifacts with more detail
        if edge_violations.get('has_edge_violations'):
            edges_str = ', '.join(edge_violations.get('edges_affected', []))
            linear_patterns = edge_violations.get('linear_patterns', {})
            
            # Report linear patterns if detected
            high_linear_edges = [edge for edge, score in linear_patterns.items() if score > 50]
            if high_linear_edges:
                diagnostics.append(f"Linear blanking patterns on: {', '.join(high_linear_edges)}")
            elif edge_violations.get('continuous_edges'):
                diagnostics.append(f"Continuous edge artifacts ({edges_str})")
            else:
                diagnostics.append(f"Edge artifacts ({edges_str})")
            
            # Add expansion recommendation if available
            if edge_violations.get('expansion_recommendations'):
                diagnostics.append("Border adjustment recommended")
            
            # Add severity note
            if edge_violations.get('severity') == 'high':
                diagnostics.append("Border detection likely missed blanking")
            elif edge_violations.get('severity') == 'medium':
                diagnostics.append("Moderate blanking detected")
        
        # Only add other diagnostics if not primarily edge issues
        if not edge_violations.get('has_edge_violations') or edge_violations.get('severity') == 'low':
            # Luma zone diagnostics
            if luma_distribution:
                primary_zone = luma_distribution.get('primary_zone')
                if primary_zone == 'highlights' and luma_distribution.get('highlight_ratio', 0) > 0.7:
                    diagnostics.append("Highlight clipping")
                elif primary_zone == 'subblack' and luma_distribution.get('subblack_ratio', 0) > 0.7:
                    diagnostics.append("Sub-black detected")
        
        return diagnostics if diagnostics else ["General broadcast range violations"]
    
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
    
    def _generate_diagnostic(self, spatial_patterns, edge_violations, luma_distribution):
        """Generate human-readable diagnostic based on pattern analysis"""
        diagnostics = []
        
        # Check for boundary artifacts FIRST (highest priority)
        if edge_violations.get('has_edge_violations'):
            edges_str = ', '.join(edge_violations.get('edges_affected', []))
            if edge_violations.get('continuous_edges'):
                diagnostics.append(f"Continuous edge artifacts ({edges_str})")
            else:
                diagnostics.append(f"Edge artifacts ({edges_str})")
            
            # Add severity note
            if edge_violations.get('severity') == 'high':
                diagnostics.append("Border detection likely missed blanking")
        
        # Only add other diagnostics if not primarily edge issues
        if not edge_violations.get('has_edge_violations') or edge_violations.get('severity') == 'low':
            # Luma zone diagnostics
            if luma_distribution:
                primary_zone = luma_distribution.get('primary_zone')
                if primary_zone == 'highlights' and luma_distribution.get('highlight_ratio', 0) > 0.7:
                    diagnostics.append("Highlight clipping")
                elif primary_zone == 'subblack' and luma_distribution.get('subblack_ratio', 0) > 0.7:
                    diagnostics.append("Sub-black detected")
        
        return diagnostics if diagnostics else ["General broadcast range violations"]
    
    def adaptive_video_sampling(self, cap_highlighted, cap_original, max_samples=100, content_start_frame=0):
        """
        Adaptively sample video based on scene changes and content complexity.
        Skips test patterns at the beginning.
        """
        logger.debug("\nPerforming adaptive video sampling...")
        logger.debug(f"  Starting from frame {content_start_frame} to skip test patterns")
        
        # Adjust total frames based on content start
        adjusted_total_frames = self.total_frames - content_start_frame
        
        # Sample distribution: more at edges, some throughout
        samples = []
        
        # Sample the beginning of actual content
        early_samples = np.linspace(content_start_frame, 
                                   content_start_frame + int(self.fps * 10), 
                                   10, dtype=int)
        samples.extend(early_samples)
        
        # Sample throughout the video
        middle_samples = np.linspace(content_start_frame + int(self.fps * 10),
                                    self.total_frames - int(self.fps * 10),
                                    max_samples - 20, dtype=int)
        samples.extend(middle_samples)
        
        # Sample the end
        end_samples = np.linspace(self.total_frames - int(self.fps * 10),
                                 self.total_frames - 1,
                                 10, dtype=int)
        samples.extend(end_samples)
        
        # Remove duplicates and sort
        samples = sorted(list(set([s for s in samples if content_start_frame <= s < self.total_frames])))[:max_samples]
        
        # Actually read the frames
        frame_samples = []
        for idx in samples:
            cap_highlighted.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_h, frame_h = cap_highlighted.read()
            
            cap_original.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_o, frame_o = cap_original.read()
            
            if ret_h and ret_o:
                frame_samples.append((idx, frame_h, frame_o))
        
        logger.debug(f"  Collected {len(frame_samples)} frame samples")
        return frame_samples
    
    def analyze_video_comprehensive(self, duration_limit=300):
        """
        Comprehensive analysis using differential detection and adaptive sampling
        """
        logger.debug(f"\nPerforming comprehensive BRNG analysis...")
        
        # Open both videos
        cap_highlighted = cv2.VideoCapture(str(self.highlighted_video))
        cap_original = cv2.VideoCapture(str(self.original_video))
        
        if not cap_highlighted.isOpened() or not cap_original.isOpened():
            logger.error("✗ Could not open processed videos")
            return None
        
        # Get video properties from highlighted version
        fps = cap_highlighted.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_highlighted.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap_highlighted.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_highlighted.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.debug(f"  Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
        
        # Find where actual content starts (skip color bars)
        cap_temp = cv2.VideoCapture(str(self.original_video))
        content_start_frame = self.find_content_start(cap_temp)
        cap_temp.release()
        
        # Adaptive sampling
        frame_samples = self.adaptive_video_sampling(cap_highlighted, cap_original, 
                                                    content_start_frame=content_start_frame)
        
        # Analyze samples
        frame_violations = []
        edge_violation_frames = []
        continuous_edge_frames = []
        linear_pattern_frames = []  # NEW: Track frames with linear patterns
        
        logger.debug(f"\nAnalyzing {len(frame_samples)} samples...")
        
        for idx, (frame_idx, frame_h, frame_o) in enumerate(frame_samples):
            # Differential detection
            violation_mask = self.detect_brng_violations_differential(frame_h, frame_o)
            violation_pixels = int(np.sum(violation_mask > 0))
            
            if violation_pixels > 0:
                # Pattern analysis with enhanced edge detection
                pattern_analysis = self.analyze_violation_patterns(violation_mask, frame_o)
                
                # Store results
                timestamp = frame_idx / fps
                frame_data = {
                    'frame': int(frame_idx),
                    'timestamp': float(timestamp),
                    'timecode': self.format_timecode(timestamp),
                    'violation_pixels': int(violation_pixels),
                    'violation_percentage': float((violation_pixels / (width * height)) * 100),
                    'pattern_analysis': pattern_analysis,
                    'diagnostics': pattern_analysis['diagnostic']
                }
                frame_violations.append(frame_data)
                
                # Track edge violations with more detail
                if pattern_analysis['boundary_artifacts']['has_edge_violations']:
                    edge_violation_frames.append(frame_idx)
                    frame_data['has_edge_violations'] = True
                    frame_data['affected_edges'] = pattern_analysis['boundary_artifacts']['edges_affected']
                    frame_data['linear_patterns'] = pattern_analysis['boundary_artifacts'].get('linear_patterns', {})
                    frame_data['expansion_recommendations'] = pattern_analysis['boundary_artifacts'].get('expansion_recommendations', {})
                    
                    if pattern_analysis['boundary_artifacts']['continuous_edges']:
                        continuous_edge_frames.append(frame_idx)
                        frame_data['has_continuous_edges'] = True
                    
                    # Track linear patterns
                    if pattern_analysis['spatial_patterns'].get('linear_patterns_detected'):
                        linear_pattern_frames.append(frame_idx)
                        frame_data['has_linear_patterns'] = True
        
        # Calculate enhanced aggregate statistics
        aggregate_summary = {}
        
        if frame_violations:
            avg_violation_pct = np.mean([f['violation_percentage'] for f in frame_violations])
            max_violation_pct = np.max([f['violation_percentage'] for f in frame_violations])
            linear_pattern_percentage = (len(linear_pattern_frames) / len(frame_samples)) * 100
            
            # Calculate edge violation statistics
            edge_violation_percentage = (len(edge_violation_frames) / len(frame_samples)) * 100
            continuous_edge_percentage = (len(continuous_edge_frames) / len(frame_samples)) * 100
            linear_pattern_percentage = (len(linear_pattern_frames) / len(frame_samples)) * 100

            # Aggregate linear pattern scores across all frames
            all_linear_patterns = {}
            for edge in ['left', 'right', 'top', 'bottom']:
                edge_scores = [f.get('linear_patterns', {}).get(edge, 0) 
                            for f in frame_violations if 'linear_patterns' in f]
                if edge_scores:
                    all_linear_patterns[edge] = np.mean(edge_scores)
            
            # Aggregate expansion recommendations
            all_expansion_recs = {}
            for f in frame_violations:
                if 'expansion_recommendations' in f:
                    for edge, rec in f['expansion_recommendations'].items():
                        if edge not in all_expansion_recs:
                            all_expansion_recs[edge] = []
                        all_expansion_recs[edge].append(rec)
            
            # Calculate consensus expansion recommendations
            consensus_expansions = {}
            for edge, recs in all_expansion_recs.items():
                if recs:
                    consensus_expansions[edge] = int(np.percentile(recs, 75))  # Use 75th percentile
            
            # Collect all affected edges
            all_affected_edges = []
            for f in frame_violations:
                if 'affected_edges' in f:
                    all_affected_edges.extend(f['affected_edges'])
            
            unique_edges = list(set(all_affected_edges))
            
            aggregate_summary = {
                'edge_violation_frames': len(edge_violation_frames),
                'edge_violation_percentage': float((len(edge_violation_frames) / len(frame_samples)) * 100),
                'continuous_edge_frames': len(continuous_edge_frames),
                'continuous_edge_percentage': float((len(continuous_edge_frames) / len(frame_samples)) * 100),
                'linear_pattern_frames': len(linear_pattern_frames),
                'linear_pattern_percentage': float(linear_pattern_percentage),
                'linear_pattern_percentages': all_linear_patterns,
                'boundary_edges_detected': list(set(all_affected_edges)),
                'consensus_expansion_recommendations': consensus_expansions,
                'edge_violation_details': {  # Pass detailed info for smart expansion
                    'linear_patterns': all_linear_patterns,
                    'expansion_recommendations': consensus_expansions,
                    'blanking_depth': {}  # Will be populated if we detect blanking depth
                },
                'requires_border_adjustment': bool(
                    linear_pattern_percentage > 20 or 
                    continuous_edge_percentage > 10 or
                    len(consensus_expansions) > 0
                )
            }
        else:
            avg_violation_pct = 0.0
            max_violation_pct = 0.0
            aggregate_summary = {
                'edge_violation_frames': 0,
                'edge_violation_percentage': 0.0,
                'continuous_edge_frames': 0,
                'continuous_edge_percentage': 0.0,
                'boundary_edges_detected': [],
                'boundary_artifact_percentage': 0.0,
                'requires_border_adjustment': False
            }

        # Separate frames into categories
        edge_only_violations = []
        content_violations = []
        mixed_violations = []
        
        for frame_data in frame_violations:
            # Check if this is primarily an edge violation
            if frame_data.get('has_edge_violations'):
                pattern_analysis = frame_data.get('pattern_analysis', {})
                edge_info = pattern_analysis.get('edge_violations', {})
                
                # Check if violations are concentrated at edges
                if edge_info.get('severity') in ['high', 'medium']:
                    # This is primarily an edge violation
                    edge_only_violations.append(frame_data)
                elif frame_data['violation_percentage'] > 0.5:
                    # Mixed - has edge violations but also significant content violations
                    mixed_violations.append(frame_data)
                else:
                    # Minor edge violations with minimal overall impact
                    edge_only_violations.append(frame_data)
            else:
                # No edge violations - this is a content violation
                content_violations.append(frame_data)
        
        # Select worst frames prioritizing content violations over edge violations
        worst_frames = []
        
        # First, add content violations (these are the real problems)
        if content_violations:
            worst_frames.extend(sorted(content_violations, 
                                    key=lambda x: x['violation_pixels'], 
                                    reverse=True)[:5])
        
        # Then add mixed violations if we need more
        if len(worst_frames) < 5 and mixed_violations:
            remaining = 5 - len(worst_frames)
            worst_frames.extend(sorted(mixed_violations, 
                                    key=lambda x: x['violation_pixels'], 
                                    reverse=True)[:remaining])
        
        # Only add edge-only violations if we have no other violations
        if len(worst_frames) == 0 and edge_only_violations:
            # But mark them specially so we know they're edge-only
            edge_frames = sorted(edge_only_violations, 
                            key=lambda x: x['violation_pixels'], 
                            reverse=True)[:3]
            for frame in edge_frames:
                frame['edge_only'] = True
            worst_frames.extend(edge_frames)
        
        # Don't save diagnostic thumbnails for edge-only violations
        saved_thumbnails = []
        if worst_frames:
            # Filter out edge-only frames from thumbnail generation
            frames_for_thumbnails = [f for f in worst_frames if not f.get('edge_only', False)]
            
            if frames_for_thumbnails:
                saved_thumbnails = self.save_diagnostic_thumbnails(
                    frames_for_thumbnails, cap_highlighted, cap_original, 
                    num_thumbnails=min(5, len(frames_for_thumbnails))
                )
            elif len(worst_frames) > 0:
                logger.info("  Skipping thumbnails - violations are only at frame edges (expected in blanking areas)")
        
        cap_highlighted.release()
        cap_original.release()
        
        # Compile comprehensive analysis
        analysis = {
            'video_info': {
                'source': str(self.video_path),
                'width': int(width),
                'height': int(height),
                'fps': float(fps),
                'total_frames': int(total_frames),
                'duration': float(total_frames / fps),
                'active_area': list(self.active_area) if self.active_area else None,
                'content_start_frame': int(content_start_frame),
                'content_start_time': float(content_start_frame / fps)
            },
            'analysis_method': 'differential_detection_improved',
            'total_frames_analyzed': len(frame_samples),
            'frames_with_violations': len(frame_violations),
            'average_violation_percentage': float(avg_violation_pct),
            'max_violation_percentage': float(max_violation_pct),
            'aggregate_patterns': aggregate_summary,
            'worst_frames': worst_frames[:5],
            'saved_thumbnails': saved_thumbnails
        }
        
        # Generate actionable report
        actionable_report = self.generate_actionable_report(analysis)
        analysis['actionable_report'] = actionable_report
        
        # Convert numpy types before saving
        analysis = self._convert_numpy_types(analysis)
        
        # Save analysis
        with open(self.analysis_output, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.debug(f"\n✔ Analysis complete. Results saved to: {self.analysis_output}\n")
        if saved_thumbnails:
            logger.info(f"✔ Saved {len(saved_thumbnails)} diagnostic thumbnail(s) to: {self.thumbnails_dir}")
        
        return analysis
    
    
    def generate_actionable_report(self, analysis_results, previous_results=None, refinement_progress=None):
        """
        Generate specific, actionable recommendations based on violation patterns.
        Now includes refinement progress analysis.
        """
        recommendations = []
        severity_score = 0
        
        aggregate = analysis_results.get('aggregate_patterns', {})
        
        # Use refinement progress if available
        if refinement_progress:
            if refinement_progress['should_continue']:
                border_adjustment_needed = True
                adjustment_reason = refinement_progress['reason']
            else:
                border_adjustment_needed = refinement_progress['recommendation'] not in ['stop_acceptable', 'stop_minimal_progress']
                adjustment_reason = refinement_progress['reason']
        else:
            # Fallback to old logic but with more nuanced thresholds
            continuous_pct = aggregate.get('continuous_edge_percentage', 0)
            edge_pct = aggregate.get('edge_violation_percentage', 0)
            
            # More sophisticated decision making
            if continuous_pct > 25:  # Very high
                border_adjustment_needed = True
                adjustment_reason = 'high_continuous_violations'
            elif continuous_pct > 15 and edge_pct > 20:  # Both moderately high
                border_adjustment_needed = True  
                adjustment_reason = 'moderate_combined_violations'
            elif continuous_pct > 10 and len(aggregate.get('boundary_edges_detected', [])) >= 3:  # Multiple edges
                border_adjustment_needed = True
                adjustment_reason = 'multiple_edge_violations'
            else:
                border_adjustment_needed = False
                adjustment_reason = 'acceptable_levels'
        
        # Generate recommendations based on analysis
        if border_adjustment_needed:
            edges = aggregate.get('boundary_edges_detected', [])
            
            if adjustment_reason == 'significant_improvement':
                recommendations.append({
                    'issue': 'Border Detection Improving - Continue Adjustment',
                    'severity': 'medium',
                    'affected_areas': f"Frame edges: {', '.join(edges)}",
                    'description': 'Border refinement is working - continue with current approach',
                    'action_required': 'Continue border detection refinement',
                    'percentage_affected': aggregate.get('continuous_edge_percentage', 0)
                })
                severity_score += 2
            elif adjustment_reason == 'acceptable_low_levels':
                recommendations.append({
                    'issue': 'Minor Edge Violations - Consider Acceptable',
                    'severity': 'low',
                    'affected_areas': f"Frame edges: {', '.join(edges)}",
                    'description': 'Low levels of edge violations may be acceptable for this content',
                    'action_required': 'Review if current quality is sufficient',
                    'percentage_affected': aggregate.get('continuous_edge_percentage', 0)
                })
                severity_score += 1
            else:
                recommendations.append({
                    'issue': 'Border Detection Needs Adjustment',
                    'severity': 'high',
                    'affected_areas': f"Frame edges: {', '.join(edges)}",
                    'description': f'Active area detection needs refinement ({adjustment_reason})',
                    'action_required': 'Re-run border detection with adjusted parameters',
                    'percentage_affected': aggregate.get('continuous_edge_percentage', 0)
                })
                severity_score += 5
        else:
            # Check for other types of violations
            total_violations = analysis_results.get('frames_with_violations', 0)
            total_analyzed = analysis_results.get('total_frames_analyzed', 1)
            violation_rate = (total_violations / total_analyzed) * 100
            
            if violation_rate > 10:
                recommendations.append({
                    'issue': 'Content-Based BRNG Violations',
                    'severity': 'medium',
                    'affected_areas': 'Video content',
                    'description': 'Violations appear to be in content rather than blanking areas',
                    'action_required': 'Review source material or encoding parameters',
                    'percentage_affected': violation_rate
                })
                severity_score += 2
        
        # Generate overall assessment with refinement context
        if refinement_progress:
            if refinement_progress['improvement_score'] > 5:
                overall_assessment = "Border refinement is working well - significant improvement detected"
                action_priority = "low" if not border_adjustment_needed else "medium"
            elif refinement_progress['improvement_score'] > 1:
                overall_assessment = "Border refinement showing modest improvement"
                action_priority = "medium"
            elif refinement_progress['reason'] == 'acceptable_low_levels':
                overall_assessment = "Video quality acceptable with current border settings"
                action_priority = "none"
            elif refinement_progress['reason'] == 'minimal_progress':
                overall_assessment = "Border refinement has reached practical limits"
                action_priority = "review"
            else:
                overall_assessment = "Border refinement not yielding expected improvements"
                action_priority = "high"
        else:
            # Fallback to original logic
            if border_adjustment_needed:
                overall_assessment = "Border detection adjustment required - edges contain violations"
                action_priority = "high"
            elif severity_score == 0:
                overall_assessment = "Video is broadcast-safe with no significant issues"
                action_priority = "none"
            elif severity_score < 3:
                overall_assessment = "Minor broadcast range issues detected"
                action_priority = "low"
            else:
                overall_assessment = "Broadcast range issues requiring attention"
                action_priority = "medium"
        
        return {
            'overall_assessment': overall_assessment,
            'action_priority': action_priority,
            'severity_score': severity_score,
            'recommendations': recommendations,
            'requires_border_adjustment': border_adjustment_needed,
            'adjustment_reason': adjustment_reason,
            'refinement_progress': refinement_progress,
            'summary_statistics': {
                'total_frames_analyzed': analysis_results.get('total_frames_analyzed', 0),
                'frames_with_violations': analysis_results.get('frames_with_violations', 0),
                'average_violation_percentage': analysis_results.get('average_violation_percentage', 0),
                'max_violation_percentage': analysis_results.get('max_violation_percentage', 0),
                'content_start_time': analysis_results.get('video_info', {}).get('content_start_time', 0)
            }
        }
    
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
    
    def save_diagnostic_thumbnails(self, worst_frames, cap_highlighted, cap_original, num_thumbnails=5):
        """
        Save diagnostic thumbnails showing violations with analysis overlay
        """
        logger.info(f"\nSaving diagnostic thumbnails for worst frames...")
        
        saved_thumbnails = []
        
        for i, frame_data in enumerate(worst_frames[:num_thumbnails]):
            frame_idx = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            # Get both frames
            cap_highlighted.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_h, frame_h = cap_highlighted.read()
            
            cap_original.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_o, frame_o = cap_original.read()
            
            if ret_h and ret_o:
                # Create diagnostic visualization with padding
                h, w = frame_h.shape[:2]
                
                # Define padding between quadrants
                padding = 10  # pixels between quadrants
                border_color = (64, 64, 64)  # Dark gray border
                
                # Create larger canvas to accommodate padding
                viz_height = h * 2 + padding
                viz_width = w * 2 + padding
                viz = np.full((viz_height, viz_width, 3), border_color, dtype=np.uint8)
                
                # Top-left: Original frame
                viz[0:h, 0:w] = frame_o
                
                # Top-right: Highlighted frame (offset by padding)
                viz[0:h, w+padding:w*2+padding] = frame_h
                
                # Bottom-left: Extract violations using differential detection (same method as main analysis)
                violation_mask = self.detect_brng_violations_differential(frame_h, frame_o)

                # Create violations-only visualization
                violations_only = np.zeros_like(frame_o)
                # Show violations as bright cyan on black background
                violations_only[violation_mask > 0] = [255, 255, 0]  # Cyan in BGR
                viz[h+padding:h*2+padding, 0:w] = violations_only
                
                # Bottom-right: Analysis information overlay (offset by padding)
                info_panel = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Add analysis info text to the info panel
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (255, 255, 255)  # White text
                thickness = 1
                line_height = 25
                
                # Prepare analysis information
                diagnostics_text = ', '.join(frame_data.get('diagnostics', ['Unknown'])[:2])
                
                # Fix the timestamp calculation to account for skip offset
                actual_timestamp = timestamp + self.skip_offset
                actual_frame = int(frame_idx + (self.skip_offset * self.fps))

                info_lines = [
                    f"Original Frame: {actual_frame}",
                    f"Original Time: {self.format_timecode(actual_timestamp)}",
                    f"Violations: {frame_data['violation_percentage']:.4f}%",
                    f"Pixels: {frame_data['violation_pixels']}",
                    f"Pattern: {diagnostics_text}"
                ]
                                
                # Add edge info if present
                if frame_data.get('has_edge_violations'):
                    edges = ', '.join(frame_data.get('affected_edges', []))
                    info_lines.append(f"Edges: {edges}")
                
                if frame_data.get('has_continuous_edges'):
                    info_lines.append("Continuous edge artifacts")
                
                # Add pattern analysis details
                pattern_analysis = frame_data.get('pattern_analysis', {})
                if pattern_analysis:
                    spatial = pattern_analysis.get('spatial_patterns', {})
                    if spatial.get('edge_concentrated'):
                        info_lines.append("Edge-concentrated violations")
                    if spatial.get('has_boundary_artifacts'):
                        severity = spatial.get('boundary_severity', 'unknown')
                        info_lines.append(f"Boundary artifacts: {severity}")
                
                # Draw text lines on info panel (centered)
                y_position = 30
                for line in info_lines:
                    # Wrap long lines if necessary
                    if len(line) > 35:  # Approximate character limit
                        words = line.split(' ')
                        current_line = ""
                        for word in words:
                            if len(current_line + " " + word) <= 35:
                                current_line += " " + word if current_line else word
                            else:
                                if current_line:
                                    # Center the text
                                    text_size = cv2.getTextSize(current_line, font, font_scale, thickness)[0]
                                    x_position = (w - text_size[0]) // 2
                                    cv2.putText(info_panel, current_line, (x_position, y_position), 
                                            font, font_scale, color, thickness)
                                    y_position += line_height
                                current_line = word
                        if current_line:
                            # Center the text
                            text_size = cv2.getTextSize(current_line, font, font_scale, thickness)[0]
                            x_position = (w - text_size[0]) // 2
                            cv2.putText(info_panel, current_line, (x_position, y_position), 
                                    font, font_scale, color, thickness)
                            y_position += line_height
                    else:
                        # Center the text
                        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                        x_position = (w - text_size[0]) // 2
                        cv2.putText(info_panel, line, (x_position, y_position), 
                                font, font_scale, color, thickness)
                        y_position += line_height
                    
                    # Stop if we're running out of space
                    if y_position > h - 30:
                        break
                
                # Add a centered title for the info panel
                title_text = "ANALYSIS DETAILS"
                title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                title_x = (w - title_size[0]) // 2
                cv2.putText(info_panel, title_text, (title_x, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Cyan title
                
                # Place info panel in bottom-right with padding
                viz[h+padding:h*2+padding, w+padding:w*2+padding] = info_panel
                
                # Add improved labels for each quadrant with better positioning
                label_font = cv2.FONT_HERSHEY_SIMPLEX
                label_scale = 1.0
                label_color = (255, 255, 255)
                label_thickness = 2
                
                # Add background rectangles for better text visibility (adjusted for padding)
                cv2.rectangle(viz, (5, 5), (200, 35), (0, 0, 0), -1)
                cv2.rectangle(viz, (w+padding+5, 5), (w+padding+200, 35), (0, 0, 0), -1)
                cv2.rectangle(viz, (5, h+padding+5), (250, h+padding+35), (0, 0, 0), -1)
                
                # Center the background rectangle for "Analysis Details" (adjusted for padding)
                analysis_label = "Analysis Details"
                analysis_label_size = cv2.getTextSize(analysis_label, label_font, label_scale, label_thickness)[0]
                analysis_bg_x = w + padding + (w - analysis_label_size[0]) // 2 - 10
                analysis_bg_width = analysis_label_size[0] + 20
                cv2.rectangle(viz, (analysis_bg_x, h+padding+5), (analysis_bg_x + analysis_bg_width, h+padding+35), (0, 0, 0), -1)
                
                # Add labels (adjusted positions for padding)
                cv2.putText(viz, "Original", (10, 25), label_font, label_scale, label_color, label_thickness)
                cv2.putText(viz, "BRNG Highlighted", (w+padding+10, 25), label_font, label_scale, label_color, label_thickness)
                cv2.putText(viz, "Violations Only", (10, h+padding+25), label_font, label_scale, label_color, label_thickness)
                
                # Center the "Analysis Details" label (adjusted for padding)
                analysis_x = w + padding + (w - analysis_label_size[0]) // 2
                cv2.putText(viz, analysis_label, (analysis_x, h+padding+25), label_font, label_scale, label_color, label_thickness)
                
                # Add crosshair lines to clearly separate quadrants
                line_color = (128, 128, 128)  # Light gray
                line_thickness = 2
                
                # Vertical divider line
                cv2.line(viz, (w + padding//2, 0), (w + padding//2, viz_height), line_color, line_thickness)
                
                # Horizontal divider line  
                cv2.line(viz, (0, h + padding//2), (viz_width, h + padding//2), line_color, line_thickness)
                
                # Save thumbnail with corrected timestamp
                # Fix the timestamp calculation here too:
                # Get skip offset from the analysis results
                skip_offset = 0
                if hasattr(self, 'skip_offset'):
                    skip_offset = self.skip_offset
                elif 'skip_info' in frame_data:
                    skip_offset = frame_data['skip_info'].get('total_skipped_seconds', 0)
                # You'll need to pass this info or store it in the analyzer
                
                actual_timestamp = timestamp + skip_offset  # Add skip offset
                timecode_str = self.format_filename_timecode(actual_timestamp)
                thumbnail_filename = f"{self.video_path.stem}_diagnostic_{timecode_str}.jpg"
                thumbnail_path = self.thumbnails_dir / thumbnail_filename
                
                cv2.imwrite(str(thumbnail_path), viz)
                saved_thumbnails.append({
                    'filename': thumbnail_filename,
                    'path': str(thumbnail_path),
                    'frame': int(frame_idx + (skip_offset * self.fps)),  # Corrected frame number
                    'timestamp': float(actual_timestamp),  # Corrected timestamp
                    'timecode': self.format_timecode(actual_timestamp)  # Corrected timecode
                })
                
                logger.info(f"  ✓ Saved: {thumbnail_filename}")
                    
                return saved_thumbnails
    
    def print_summary(self, analysis):
        """Print enhanced analysis summary"""
        logger.debug("\n" + "="*80)
        logger.debug("ACTIVE AREA BRNG ANALYSIS RESULTS - IMPROVED")
        logger.debug("="*80)
        
        info = analysis['video_info']
        
        if info['active_area']:
            logger.debug(f"Active Area: {info['active_area'][2]}x{info['active_area'][3]} at ({info['active_area'][0]},{info['active_area'][1]})")
        else:
            logger.debug("Analyzed: Full frame (no border data)")
        
        logger.debug(f"Video: {info['width']}x{info['height']}, {info['duration']:.1f}s")
        
        # Note about skipped content
        if info.get('content_start_time', 0) > 0:
            logger.debug(f"Skipped test patterns: Started analysis at {info['content_start_time']:.1f}s")
        
        logger.debug(f"Frames analyzed: {analysis['total_frames_analyzed']}")
        
        report = analysis.get('actionable_report', {})
        
        logger.debug(f"\n{report.get('overall_assessment', 'Analysis complete')}\n")
        logger.debug(f"Priority: {report.get('action_priority', 'none').upper()}")
        
        # HIGHLIGHT EDGE VIOLATIONS
        aggregate = analysis.get('aggregate_patterns', {})
        if aggregate.get('requires_border_adjustment'):
            logger.debug("\n⚠️  BORDER ADJUSTMENT REQUIRED")
            logger.debug(f"  Continuous edge violations detected on: {', '.join(aggregate.get('boundary_edges_detected', []))}")
            logger.debug(f"  {aggregate.get('continuous_edge_percentage', 0):.1f}% of frames have continuous edge artifacts")
            logger.debug("  → Re-run border detection with adjusted parameters")
        elif aggregate.get('edge_violation_percentage', 0) > 0:
            logger.debug(f"\nEdge violations: {aggregate['edge_violation_percentage']:.1f}% of frames")
            logger.debug(f"  Affected edges: {', '.join(aggregate.get('boundary_edges_detected', []))}")
        
        if analysis['frames_with_violations'] > 0:
            logger.debug(f"\nVIOLATION STATISTICS:")
            logger.debug(f"  Frames with violations: {analysis['frames_with_violations']}/{analysis['total_frames_analyzed']}")
            logger.debug(f"  Average violation: {analysis['average_violation_percentage']:.4f}% of pixels")
            logger.debug(f"  Maximum violation: {analysis['max_violation_percentage']:.4f}% of pixels")
        
        # Worst frames (if not all edge artifacts)
        if analysis.get('worst_frames'):
            # Check if all worst frames are edge-only
            all_edge_only = all(f.get('edge_only', False) for f in analysis['worst_frames'])
            
            if all_edge_only:
                logger.info("\nEDGE VIOLATIONS ONLY:")
                logger.info("  All violations occur at frame edges (blanking areas)")
                logger.info("  This is expected behavior for analog video with blanking")
                logger.info("  Border detection successfully identified active area")
            else:
                logger.info("\nWORST FRAMES (Content Violations):")
                for i, frame in enumerate(analysis['worst_frames'][:3], 1):
                    if not frame.get('edge_only', False):
                        edge_note = " [MIXED]" if frame.get('has_edge_violations') else ""
                        logger.info(f"  {i}. Frame {frame['frame']} ({frame['timecode']}) - {frame['violation_percentage']:.4f}% pixels{edge_note}")
                        if frame.get('diagnostics'):
                            logger.info(f"     Issues: {', '.join(frame['diagnostics'][:2])}")
        
        logger.debug("="*80)
    
    def cleanup(self, keep_intermediates=False):
        """Clean up temporary files"""
        try:
            if not keep_intermediates:
                if self.highlighted_video.exists():
                    self.highlighted_video.unlink()
                if self.original_video.exists():
                    self.original_video.unlink()
                logger.debug("\n✔ Removed intermediate videos")
            else:
                logger.debug(f"ℹ️ Kept intermediate videos in: {self.temp_dir}")
            
            # Remove temp directory if empty
            if self.temp_dir.exists():
                if not any(self.temp_dir.iterdir()):
                    self.temp_dir.rmdir()
                    logger.debug("✔ Cleaned up temporary directory")
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up temp files: {e}")

    
    def analyze_refinement_progress(self, current_results, previous_results=None):
        """
        Analyze if border refinement is making meaningful progress
        Returns recommendation for whether to continue refining
        """
        if not previous_results:
            return {
                'should_continue': True,
                'reason': 'first_attempt',
                'improvement_score': 0,
                'recommendation': 'continue'
            }
        
        current_agg = current_results.get('aggregate_patterns', {})
        previous_agg = previous_results.get('aggregate_patterns', {})
        
        # Extract metrics
        curr_edge_pct = current_agg.get('edge_violation_percentage', 0)
        prev_edge_pct = previous_agg.get('edge_violation_percentage', 0)
        
        curr_linear_patterns = current_agg.get('linear_pattern_percentages', {})
        prev_linear_patterns = previous_agg.get('linear_pattern_percentages', {})
        
        curr_worst_frames = current_results.get('worst_frames', [])
        prev_worst_frames = previous_results.get('worst_frames', [])
        
        curr_worst_pixels = curr_worst_frames[0]['violation_percentage'] if curr_worst_frames else 0
        prev_worst_pixels = prev_worst_frames[0]['violation_percentage'] if prev_worst_frames else 0
        
        # Calculate improvements
        edge_improvement = prev_edge_pct - curr_edge_pct
        worst_pixels_improvement = prev_worst_pixels - curr_worst_pixels
        
        # Check if we're stuck in a pattern
        if abs(worst_pixels_improvement) < 0.001 and abs(edge_improvement) < 1:
            # Almost no change - we're stuck
            return {
                'should_continue': False,
                'reason': 'no_progress',
                'improvement_score': 0,
                'recommendation': 'stop_no_progress'
            }
        
        # Check if we've achieved good quality
        if curr_worst_pixels < 0.01 and curr_edge_pct < 5:
            return {
                'should_continue': False,
                'reason': 'excellent_quality_achieved',
                'improvement_score': 100,
                'recommendation': 'stop_excellent'
            }
        
        if curr_worst_pixels < 0.1 and curr_edge_pct < 10:
            return {
                'should_continue': False,
                'reason': 'acceptable_quality_achieved',
                'improvement_score': 50,
                'recommendation': 'stop_acceptable'
            }
        
        # Check if we're making meaningful progress
        if worst_pixels_improvement > 0.5 or edge_improvement > 10:
            return {
                'should_continue': True,
                'reason': 'significant_improvement',
                'improvement_score': worst_pixels_improvement * 100 + edge_improvement,
                'recommendation': 'continue'
            }
        
        # Check if linear patterns are still present
        max_linear = max(curr_linear_patterns.values(), default=0) if curr_linear_patterns else 0
        if max_linear > 40:
            return {
                'should_continue': True,
                'reason': 'linear_patterns_present',
                'improvement_score': worst_pixels_improvement * 50,
                'recommendation': 'continue_targeted'
            }
        
        # Default: stop if we're not making progress
        return {
            'should_continue': False,
            'reason': 'minimal_progress',
            'improvement_score': worst_pixels_improvement * 10,
            'recommendation': 'stop_minimal_progress'
        }

def analyze_active_area_brng(video_path, border_data_path=None, output_dir=None, 
                            duration_limit=300, skip_start_seconds=None):
    """
    Main function to analyze BRNG violations with improved detection.
    
    Args:
        video_path: Path to video file
        border_data_path: Path to border detection JSON
        output_dir: Output directory for results
        duration_limit: Maximum seconds to process
        skip_start_seconds: Skip this many seconds from start (e.g., for color bars)
    
    Returns:
        Analysis results dictionary
    """
    analyzer = ActiveAreaBrngAnalyzer(video_path, border_data_path, output_dir)
    
    # Store the skip offset in the analyzer instance
    analyzer.skip_offset = skip_start_seconds if skip_start_seconds else 0
    
    # Determine skip time - use the greater of color bars end or content start detection
    skip_seconds = 0
    
    # First check if we should skip color bars
    if skip_start_seconds:
        skip_seconds = skip_start_seconds
        logger.info(f"Will skip first {skip_seconds:.1f}s based on color bars detection\n")
    
    # Also check for test patterns (but only if not already skipping past them)
    cap = cv2.VideoCapture(str(video_path))
    content_start_frame = analyzer.find_content_start(cap, max_seconds=30)
    content_start_seconds = content_start_frame / analyzer.fps if analyzer.fps > 0 else 0
    cap.release()
    
    # Use the greater of the two skip times
    if content_start_seconds > skip_seconds:
        skip_seconds = content_start_seconds
        logger.info(f"Additional test patterns detected, extending skip to {skip_seconds:.1f}s")
    
    # Process with ffmpeg, skipping the determined amount
    if not analyzer.process_with_ffmpeg(duration_limit, skip_start_seconds=skip_seconds):
        logger.error("FFmpeg processing failed")
        return None
    
    # Comprehensive analysis
    analysis = analyzer.analyze_video_comprehensive(duration_limit)
    
    if analysis:
        # Add information about what was skipped
        analysis['skip_info'] = {
            'total_skipped_seconds': skip_seconds,
            'color_bars_skip': skip_start_seconds if skip_start_seconds else 0,
            'test_pattern_skip': content_start_seconds
        }
        analyzer.print_summary(analysis)
    
    # Clean up temp files
    analyzer.cleanup(keep_intermediates=False)
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze BRNG violations with improved edge detection'
    )
    parser.add_argument('video_file', help='Path to video file')
    parser.add_argument('--border-data', help='Path to border detection JSON file')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--duration', type=int, default=300, 
                       help='Max duration to analyze (seconds, default: 300)')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep intermediate files for debugging')
    
    args = parser.parse_args()
    
    results = analyze_active_area_brng(
        args.video_file,
        args.border_data,
        args.output_dir,
        args.duration
    )