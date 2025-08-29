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
                logger.debug(f"✔ Loaded border data. Active area: {self.active_area}")
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
        Specifically check for violations at the very edges of the frame
        Returns detailed information about edge violations
        """
        h, w = violation_mask.shape
        edge_info = {
            'has_edge_violations': False,
            'edges_affected': [],
            'edge_percentages': {},
            'continuous_edges': [],
            'severity': 'none'
        }
        
        # Check each edge
        edges_to_check = [
            ('left', violation_mask[:, :edge_width]),
            ('right', violation_mask[:, -edge_width:]),
            ('top', violation_mask[:edge_width, :]),
            ('bottom', violation_mask[-edge_width:, :])
        ]
        
        for edge_name, edge_region in edges_to_check:
            if edge_region.size == 0:
                continue
            
            violation_pixels = np.sum(edge_region > 0)
            total_pixels = edge_region.size
            violation_percentage = (violation_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            edge_info['edge_percentages'][edge_name] = violation_percentage
            
            if violation_percentage > 10:  # More than 10% of edge has violations
                edge_info['edges_affected'].append(edge_name)
                edge_info['has_edge_violations'] = True
                
                # Check if it's a continuous line
                if edge_name in ['left', 'right']:
                    # Check vertical continuity
                    edge_column = violation_mask[:, 0] if edge_name == 'left' else violation_mask[:, -1]
                    continuity = np.sum(edge_column > 0) / h
                    if continuity > 0.7:  # 70% of the edge height
                        edge_info['continuous_edges'].append(edge_name)
                else:
                    # Check horizontal continuity
                    edge_row = violation_mask[0, :] if edge_name == 'top' else violation_mask[-1, :]
                    continuity = np.sum(edge_row > 0) / w
                    if continuity > 0.7:  # 70% of the edge width
                        edge_info['continuous_edges'].append(edge_name)
        
        # Determine severity
        if len(edge_info['continuous_edges']) >= 2:
            edge_info['severity'] = 'high'
        elif len(edge_info['continuous_edges']) >= 1:
            edge_info['severity'] = 'medium'
        elif len(edge_info['edges_affected']) >= 2:
            edge_info['severity'] = 'low'
        
        return edge_info
    
    def analyze_violation_patterns(self, violation_mask, frame):
        """
        Analyze BRNG violations with emphasis on edge detection
        """
        h, w = violation_mask.shape
        
        # FIRST: Check for edge violations (highest priority)
        edge_violations = self.detect_edge_violations(violation_mask)
        
        # Edge-based analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check if violations occur at edges
        edge_violations_mask = cv2.bitwise_and(violation_mask, edges)
        edge_violation_ratio = np.sum(edge_violations_mask > 0) / max(1, np.sum(violation_mask > 0))
        
        # Spatial distribution
        spatial_patterns = {
            'edge_concentrated': bool(edge_violation_ratio > 0.6),
            'has_boundary_artifacts': edge_violations['has_edge_violations'],
            'boundary_edges': edge_violations['edges_affected'],
            'boundary_severity': edge_violations['severity']
        }
        
        # Analyze in context of luma zones (but not if edge violations)
        luma_distribution = {}
        if not edge_violations['has_edge_violations']:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luma_distribution = self._analyze_luma_zone_violations(violation_mask, gray)
        
        # Generate diagnostic
        diagnostic = self._generate_diagnostic(spatial_patterns, edge_violations, luma_distribution)
        
        return {
            'spatial_patterns': spatial_patterns,
            'edge_violations': edge_violations,
            'luma_distribution': luma_distribution,
            'edge_violation_ratio': float(edge_violation_ratio),
            'diagnostic': diagnostic,
            'boundary_artifacts': edge_violations  # Make this easily accessible
        }
    
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
        
        logger.debug(f"\nAnalyzing {len(frame_samples)} samples...")
        
        for idx, (frame_idx, frame_h, frame_o) in enumerate(frame_samples):
            # Differential detection
            violation_mask = self.detect_brng_violations_differential(frame_h, frame_o)
            violation_pixels = int(np.sum(violation_mask > 0))
            
            if violation_pixels > 0:
                # Pattern analysis with edge detection
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
                
                # Track edge violations specifically
                if pattern_analysis['boundary_artifacts']['has_edge_violations']:
                    edge_violation_frames.append(frame_idx)
                    frame_data['has_edge_violations'] = True
                    frame_data['affected_edges'] = pattern_analysis['boundary_artifacts']['edges_affected']
                    
                    if pattern_analysis['boundary_artifacts']['continuous_edges']:
                        continuous_edge_frames.append(frame_idx)
                        frame_data['has_continuous_edges'] = True
            
            # Progress indicator
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(frame_samples)} samples")
        
        # Calculate aggregate statistics with emphasis on edge violations
        aggregate_summary = {}
        
        if frame_violations:
            avg_violation_pct = np.mean([f['violation_percentage'] for f in frame_violations])
            max_violation_pct = np.max([f['violation_percentage'] for f in frame_violations])
            
            # Calculate edge violation statistics
            edge_violation_percentage = (len(edge_violation_frames) / len(frame_samples)) * 100
            continuous_edge_percentage = (len(continuous_edge_frames) / len(frame_samples)) * 100
            
            # Collect all affected edges
            all_affected_edges = []
            for f in frame_violations:
                if 'affected_edges' in f:
                    all_affected_edges.extend(f['affected_edges'])
            
            unique_edges = list(set(all_affected_edges))
            
            aggregate_summary = {
                'edge_violation_frames': len(edge_violation_frames),
                'edge_violation_percentage': float(edge_violation_percentage),
                'continuous_edge_frames': len(continuous_edge_frames),
                'continuous_edge_percentage': float(continuous_edge_percentage),
                'boundary_edges_detected': unique_edges,
                'boundary_artifact_percentage': float(edge_violation_percentage),  # For compatibility
                'requires_border_adjustment': bool(continuous_edge_percentage > 10)  # Clear signal
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
        
        # Find worst frames (excluding those that are just edge artifacts)
        non_edge_violations = [f for f in frame_violations if not f.get('has_continuous_edges', False)]
        if non_edge_violations:
            worst_frames = sorted(non_edge_violations, 
                                key=lambda x: x['violation_pixels'], 
                                reverse=True)[:10]
        else:
            worst_frames = sorted(frame_violations, 
                                key=lambda x: x['violation_pixels'], 
                                reverse=True)[:10]
        
        # Save diagnostic thumbnails for worst non-edge frames
        saved_thumbnails = []
        if worst_frames:
            saved_thumbnails = self.save_diagnostic_thumbnails(
                worst_frames, cap_highlighted, cap_original, num_thumbnails=min(5, len(worst_frames))
            )
        
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
        
        logger.debug(f"✔ Analysis complete. Results saved to: {self.analysis_output}")
        if saved_thumbnails:
            logger.info(f"✔ Saved {len(saved_thumbnails)} diagnostic thumbnail(s) to: {self.thumbnails_dir}")
        
        return analysis
    
    def generate_actionable_report(self, analysis_results):
        """
        Generate specific, actionable recommendations based on violation patterns.
        """
        recommendations = []
        severity_score = 0
        
        aggregate = analysis_results.get('aggregate_patterns', {})
        
        # Check for boundary artifacts FIRST (highest priority)
        if aggregate.get('requires_border_adjustment'):
            edges = aggregate.get('boundary_edges_detected', [])
            recommendations.append({
                'issue': 'Border Detection Needs Adjustment',
                'severity': 'high',
                'affected_areas': f"Frame edges: {', '.join(edges)}",
                'description': 'Active area detection missed blanking/border regions',
                'action_required': 'Re-run border detection with adjusted parameters',
                'percentage_affected': aggregate.get('continuous_edge_percentage', 0)
            })
            severity_score += 5  # High severity to trigger re-detection
        elif aggregate.get('edge_violation_percentage', 0) > 5:
            # Minor edge violations but not continuous
            recommendations.append({
                'issue': 'Minor Edge Artifacts',
                'severity': 'low',
                'affected_areas': f"Frame edges: {', '.join(aggregate.get('boundary_edges_detected', []))}",
                'description': 'Occasional edge violations, may be acceptable',
                'percentage_affected': aggregate.get('edge_violation_percentage', 0)
            })
            severity_score += 1
        
        # Generate overall assessment
        if aggregate.get('requires_border_adjustment'):
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
                # Create diagnostic visualization
                h, w = frame_h.shape[:2]
                
                # Create 2x2 grid visualization
                viz = np.zeros((h*2, w*2, 3), dtype=np.uint8)
                
                # Top-left: Original frame
                viz[:h, :w] = frame_o
                
                # Top-right: Highlighted frame
                viz[:h, w:] = frame_h
                
                # Bottom-left: Extract cyan-highlighted pixels from the highlighted frame
                # Create a mask for cyan pixels in the highlighted frame
                frame_h_hsv = cv2.cvtColor(frame_h, cv2.COLOR_BGR2HSV)
                cyan_mask = cv2.inRange(frame_h_hsv, 
                                    np.array([80, 100, 100]),   # Lower cyan threshold
                                    np.array([110, 255, 255]))  # Upper cyan threshold
                
                # Create violations-only visualization
                violations_only = np.zeros_like(frame_o)
                # Show cyan violations as bright cyan on black background
                violations_only[cyan_mask > 0] = [255, 255, 0]  # Cyan in BGR
                viz[h:, :w] = violations_only
                
                # Bottom-right: Analysis information overlay
                info_panel = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Add analysis info text to the info panel
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (255, 255, 255)  # White text
                thickness = 1
                line_height = 25
                
                # Prepare analysis information
                diagnostics_text = ', '.join(frame_data.get('diagnostics', ['Unknown'])[:2])
                info_lines = [
                    f"Frame: {frame_idx}",
                    f"Time: {self.format_timecode(timestamp)}",
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
                
                viz[h:, w:] = info_panel
                
                # Add labels for each quadrant
                label_font = cv2.FONT_HERSHEY_SIMPLEX
                label_scale = 1.0
                label_color = (255, 255, 255)
                label_thickness = 2
                
                # Add background rectangles for better text visibility
                cv2.rectangle(viz, (5, 5), (200, 35), (0, 0, 0), -1)
                cv2.rectangle(viz, (w+5, 5), (w+200, 35), (0, 0, 0), -1)
                cv2.rectangle(viz, (5, h+5), (250, h+35), (0, 0, 0), -1)
                # Center the background rectangle for "Analysis Details"
                analysis_label = "Analysis Details"
                analysis_label_size = cv2.getTextSize(analysis_label, label_font, label_scale, label_thickness)[0]
                analysis_bg_x = w + (w - analysis_label_size[0]) // 2 - 10
                analysis_bg_width = analysis_label_size[0] + 20
                cv2.rectangle(viz, (analysis_bg_x, h+5), (analysis_bg_x + analysis_bg_width, h+35), (0, 0, 0), -1)
                
                cv2.putText(viz, "Original", (10, 25), label_font, label_scale, label_color, label_thickness)
                cv2.putText(viz, "BRNG Highlighted", (w+10, 25), label_font, label_scale, label_color, label_thickness)
                cv2.putText(viz, "Violations Only", (10, h+25), label_font, label_scale, label_color, label_thickness)
                # Center the "Analysis Details" label
                analysis_x = w + (w - analysis_label_size[0]) // 2
                cv2.putText(viz, analysis_label, (analysis_x, h+25), label_font, label_scale, label_color, label_thickness)
                
                # Save thumbnail
                timecode_str = self.format_filename_timecode(timestamp)
                thumbnail_filename = f"{self.video_path.stem}_diagnostic_{timecode_str}.jpg"
                thumbnail_path = self.thumbnails_dir / thumbnail_filename
                
                cv2.imwrite(str(thumbnail_path), viz)
                saved_thumbnails.append({
                    'filename': thumbnail_filename,
                    'path': str(thumbnail_path),
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'timecode': self.format_timecode(timestamp)
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
            logger.info("\nWORST FRAMES:")
            for i, frame in enumerate(analysis['worst_frames'][:3], 1):
                edge_note = " [EDGE]" if frame.get('has_edge_violations') else ""
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