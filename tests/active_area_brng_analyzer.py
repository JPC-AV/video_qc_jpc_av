#!/usr/bin/env python3
"""
Active Area BRNG Analyzer - Revised Version with Enhanced Analysis

Analyzes broadcast range violations specifically in the active picture area,
using differential analysis to eliminate false positives and providing
meaningful video-specific pattern recognition.

Key improvements:
- Differential analysis to eliminate false cyan content detection
- Video-specific spatial pattern analysis
- Adaptive sampling based on scene changes
- Temporal consistency analysis
- Actionable reporting with specific issue identification
"""

import subprocess
import cv2
import numpy as np
import json
from pathlib import Path
import sys
from collections import defaultdict
from scipy import ndimage, signal
from scipy.interpolate import interp1d
import shlex


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
                print(f"✔ Loaded border data. Active area: {self.active_area}")
            else:
                print("⚠️ Border data doesn't contain active area")
        except Exception as e:
            print(f"⚠️ Could not load border data: {e}")
    
    def process_with_ffmpeg(self, duration_limit=300):
        """
        Process video with ffmpeg signalstats, creating both highlighted and original versions
        for differential analysis.
        """
        print(f"\nProcessing video with ffmpeg signalstats...")
        
        # Build filter chain for active area crop
        crop_filter = ""
        if self.active_area:
            x, y, w, h = self.active_area
            crop_filter = f"crop={w}:{h}:{x}:{y},"
            print(f"  Cropping to active area: {w}x{h} at ({x},{y})")
        else:
            print("  No border data - analyzing full frame")
        
        # Create highlighted version with BRNG violations marked in cyan
        highlighted_cmd = [
            "ffmpeg",
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
            "-i", str(self.video_path),
            "-t", str(duration_limit),
            "-vf", crop_filter.rstrip(',') if crop_filter else "null",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            str(self.original_video)
        ]
        
        print(f"  Processing up to {duration_limit} seconds...")
        
        try:
            # Process highlighted version
            print("  Creating highlighted version...")
            result = subprocess.run(highlighted_cmd, capture_output=True, text=True, check=True)
            
            # Process original version
            print("  Creating original version for comparison...")
            result = subprocess.run(original_cmd, capture_output=True, text=True, check=True)
            
            print(f"✔ FFmpeg processing complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg error: {e}")
            if e.stderr:
                print(f"  Error details: {e.stderr[:500]}")
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
    
    def _detect_scene_changes(self, cap, stride=60, threshold=30):
        """
        Detect scene boundaries for adaptive sampling
        """
        scene_boundaries = []
        prev_hist = None
        current_start = 0
        
        for frame_idx in range(0, self.total_frames, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = hist.flatten() / np.sum(hist)
            
            if prev_hist is not None:
                # Calculate histogram difference
                diff = np.sum(np.abs(hist - prev_hist))
                
                if diff > threshold / 100.0:  # Scene change detected
                    scene_boundaries.append((current_start, frame_idx))
                    current_start = frame_idx
            
            prev_hist = hist
        
        # Add final scene
        scene_boundaries.append((current_start, self.total_frames - 1))
        
        return scene_boundaries
    
    def adaptive_video_sampling(self, cap_highlighted, cap_original, max_samples=100, min_samples=20):
        """
        Adaptively sample video based on scene changes and content complexity.
        Returns list of (frame_idx, highlighted_frame, original_frame) tuples.
        """
        print("\nPerforming adaptive video sampling...")
        
        # Phase 1: Quick scene detection
        scene_boundaries = self._detect_scene_changes(cap_highlighted, stride=int(self.fps * 2))
        print(f"  Detected {len(scene_boundaries)} scenes")
        
        # Phase 2: Sample each scene proportionally
        samples = []
        samples_per_scene = max(2, min_samples // max(1, len(scene_boundaries)))
        
        for scene_start, scene_end in scene_boundaries:
            scene_duration = scene_end - scene_start
            
            # More samples for longer scenes
            scene_samples = min(
                samples_per_scene * 2,
                max(2, int(samples_per_scene * (scene_duration / self.total_frames * 2)))
            )
            
            # Sample within scene
            scene_frame_indices = np.linspace(scene_start, scene_end - 1, scene_samples, dtype=int)
            samples.extend(scene_frame_indices)
        
        # Phase 3: Add samples at potential problem areas
        # Sample more at beginning/end where analog artifacts are common
        problem_areas = [
            0, 1, 2,  # Start
            self.total_frames - 3, self.total_frames - 2, self.total_frames - 1,  # End
            int(self.fps * 10), int(self.fps * 20)  # Early content
        ]
        samples.extend([f for f in problem_areas if 0 <= f < self.total_frames])
        
        # Remove duplicates and sort
        samples = sorted(list(set(samples)))[:max_samples]
        
        # Phase 4: Actually read the frames
        frame_samples = []
        for idx in samples:
            cap_highlighted.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_h, frame_h = cap_highlighted.read()
            
            cap_original.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_o, frame_o = cap_original.read()
            
            if ret_h and ret_o:
                frame_samples.append((idx, frame_h, frame_o))
        
        print(f"  Collected {len(frame_samples)} frame samples across {len(scene_boundaries)} scenes")
        return frame_samples
    
    def _check_perimeter_concentration(self, mask, border_width=20):
        """Check if violations are concentrated at frame perimeter"""
        h, w = mask.shape
        
        # Define perimeter region
        perimeter_mask = np.zeros_like(mask)
        perimeter_mask[:border_width, :] = 1  # Top
        perimeter_mask[-border_width:, :] = 1  # Bottom
        perimeter_mask[:, :border_width] = 1  # Left
        perimeter_mask[:, -border_width:] = 1  # Right
        
        perimeter_violations = np.sum(mask & perimeter_mask)
        total_violations = np.sum(mask)
        
        if total_violations == 0:
            return False
        
        return (perimeter_violations / total_violations) > 0.5
    
    def _calculate_scatter_score(self, mask):
        """Calculate how scattered violations are (0=clustered, 1=scattered)"""
        if np.sum(mask) == 0:
            return 0.0
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        if num_labels <= 1:
            return 0.0
        
        # Calculate average distance between components
        centroids = []
        for label in range(1, num_labels):
            y_coords, x_coords = np.where(labels == label)
            if len(y_coords) > 0:
                centroids.append((np.mean(y_coords), np.mean(x_coords)))
        
        if len(centroids) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                              (centroids[i][1] - centroids[j][1])**2)
                distances.append(dist)
        
        # Normalize by image diagonal
        diagonal = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
        avg_distance = np.mean(distances) / diagonal
        
        return min(1.0, avg_distance * 3)  # Scale to 0-1 range
    
    def _detect_periodicity(self, fft_data, threshold=0.1):
        """Detect periodic patterns in FFT data"""
        magnitude = np.abs(fft_data)
        magnitude[0] = 0  # Remove DC component
        
        # Find peaks
        peaks = signal.find_peaks(magnitude, height=np.max(magnitude) * threshold)[0]
        
        if len(peaks) > 1:
            # Check for regular spacing
            peak_spacing = np.diff(peaks)
            if np.std(peak_spacing) / (np.mean(peak_spacing) + 1e-6) < 0.3:
                return True
        
        return False
    
    def _detect_block_patterns(self, mask, block_size=8):
        """Detect macroblock-like patterns"""
        h, w = mask.shape
        
        # Divide into blocks and check for block boundaries
        block_edges = np.zeros_like(mask)
        
        for y in range(block_size, h, block_size):
            block_edges[y-1:y+1, :] = 1
        for x in range(block_size, w, block_size):
            block_edges[:, x-1:x+1] = 1
        
        # Check if violations align with block boundaries
        block_aligned = np.sum(mask & block_edges)
        total_violations = np.sum(mask)
        
        if total_violations == 0:
            return False
        
        return (block_aligned / total_violations) > 0.3
    
    def _detect_boundary_artifacts(self, mask, edge_threshold=5, continuity_threshold=0.7):
        """
        Detect if violations form continuous lines at frame boundaries.
        These often indicate missed border/blanking areas.
        
        Args:
            mask: Binary violation mask
            edge_threshold: Maximum distance from edge to consider (pixels)
            continuity_threshold: Minimum fraction of edge that must have violations
        
        Returns:
            dict: Information about detected boundary artifacts
        """
        h, w = mask.shape
        results = {
            'left_edge': False,
            'right_edge': False,
            'top_edge': False,
            'bottom_edge': False,
            'boundary_width': edge_threshold
        }
        
        # Check left edge
        left_strip = mask[:, :edge_threshold]
        if left_strip.size > 0:
            # Check if violations form a continuous vertical line
            left_coverage = np.sum(np.any(left_strip, axis=1)) / h
            if left_coverage > continuity_threshold:
                # Check if violations are concentrated at the very edge
                edge_column_violations = np.sum(mask[:, 0]) / h
                if edge_column_violations > 0.5 or left_coverage > 0.8:
                    results['left_edge'] = True
        
        # Check right edge
        right_strip = mask[:, -edge_threshold:]
        if right_strip.size > 0:
            right_coverage = np.sum(np.any(right_strip, axis=1)) / h
            if right_coverage > continuity_threshold:
                edge_column_violations = np.sum(mask[:, -1]) / h
                if edge_column_violations > 0.5 or right_coverage > 0.8:
                    results['right_edge'] = True
        
        # Check top edge
        top_strip = mask[:edge_threshold, :]
        if top_strip.size > 0:
            top_coverage = np.sum(np.any(top_strip, axis=0)) / w
            if top_coverage > continuity_threshold:
                edge_row_violations = np.sum(mask[0, :]) / w
                if edge_row_violations > 0.5 or top_coverage > 0.8:
                    results['top_edge'] = True
        
        # Check bottom edge
        bottom_strip = mask[-edge_threshold:, :]
        if bottom_strip.size > 0:
            bottom_coverage = np.sum(np.any(bottom_strip, axis=0)) / w
            if bottom_coverage > continuity_threshold:
                edge_row_violations = np.sum(mask[-1, :]) / w
                if edge_row_violations > 0.5 or bottom_coverage > 0.8:
                    results['bottom_edge'] = True
        
        # Calculate detailed metrics for any detected edges
        if any([results['left_edge'], results['right_edge'], results['top_edge'], results['bottom_edge']]):
            edge_details = []
            
            if results['left_edge']:
                # Find the width of the artifact
                for x in range(min(20, w)):
                    column_coverage = np.sum(mask[:, x]) / h
                    if column_coverage < 0.3:
                        edge_details.append(('left', x))
                        break
            
            if results['right_edge']:
                for x in range(min(20, w)):
                    column_coverage = np.sum(mask[:, -(x+1)]) / h
                    if column_coverage < 0.3:
                        edge_details.append(('right', x))
                        break
            
            if results['top_edge']:
                for y in range(min(20, h)):
                    row_coverage = np.sum(mask[y, :]) / w
                    if row_coverage < 0.3:
                        edge_details.append(('top', y))
                        break
            
            if results['bottom_edge']:
                for y in range(min(20, h)):
                    row_coverage = np.sum(mask[-(y+1), :]) / w
                    if row_coverage < 0.3:
                        edge_details.append(('bottom', y))
                        break
            
            results['edge_widths'] = edge_details
        
        return results
    
    def _get_primary_zone(self, shadow, midtone, highlight):
        """Determine primary brightness zone for violations"""
        zones = {'shadows': shadow, 'midtones': midtone, 'highlights': highlight}
        return max(zones, key=zones.get)
    
    def analyze_violation_patterns(self, violation_mask, frame):
        """
        Analyze BRNG violations in context of video characteristics.
        """
        h, w = violation_mask.shape
        
        # 1. Edge-based analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check if violations occur at edges
        edge_violations = cv2.bitwise_and(violation_mask, edges)
        edge_violation_ratio = np.sum(edge_violations > 0) / max(1, np.sum(violation_mask > 0))
        
        # 2. Analyze spatial distribution
        spatial_patterns = {
            'edge_concentrated': bool(edge_violation_ratio > 0.6),
            'top_heavy': bool(np.sum(violation_mask[:h//3, :]) > np.sum(violation_mask[2*h//3:, :]) * 2),
            'bottom_heavy': bool(np.sum(violation_mask[2*h//3:, :]) > np.sum(violation_mask[:h//3, :]) * 2),
            'left_biased': bool(np.sum(violation_mask[:, :w//3]) > np.sum(violation_mask[:, 2*w//3:]) * 2),
            'right_biased': bool(np.sum(violation_mask[:, 2*w//3:]) > np.sum(violation_mask[:, :w//3]) * 2),
            'perimeter_concentrated': bool(self._check_perimeter_concentration(violation_mask)),
            'scattered': bool(self._calculate_scatter_score(violation_mask) > 0.7)
        }
        
        # 3. Check for systematic patterns
        systematic_patterns = self._detect_systematic_patterns(violation_mask)
        
        # 4. Analyze in context of luma zones
        luma_distribution = self._analyze_luma_zone_violations(violation_mask, gray)
        
        # 5. Generate diagnostic
        diagnostic = self._generate_diagnostic(spatial_patterns, systematic_patterns, luma_distribution)
        
        return {
            'spatial_patterns': spatial_patterns,
            'systematic_patterns': systematic_patterns,
            'luma_distribution': luma_distribution,
            'edge_violation_ratio': float(edge_violation_ratio),
            'diagnostic': diagnostic
        }
    
    def _detect_systematic_patterns(self, mask):
        """
        Detect systematic patterns like scan lines, macroblocks, etc.
        """
        h, w = mask.shape
        patterns = {}
        
        # Check for horizontal banding (common in interlaced content)
        row_sums = np.sum(mask, axis=1)
        if len(row_sums) > 10:
            row_fft = np.fft.fft(row_sums)
            patterns['horizontal_banding'] = self._detect_periodicity(row_fft)
        else:
            patterns['horizontal_banding'] = False
        
        # Check for vertical banding
        col_sums = np.sum(mask, axis=0)
        if len(col_sums) > 10:
            col_fft = np.fft.fft(col_sums)
            patterns['vertical_banding'] = self._detect_periodicity(col_fft)
        else:
            patterns['vertical_banding'] = False
        
        # Check for macroblock artifacts (8x8 or 16x16 patterns)
        patterns['macroblock_artifacts'] = self._detect_block_patterns(mask)
        
        return patterns
    
    def _analyze_luma_zone_violations(self, mask, gray_frame):
        """
        Analyze where violations occur in terms of brightness zones.
        """
        # Define luma zones
        shadows = (gray_frame < 64)
        midtones = (gray_frame >= 64) & (gray_frame < 192)
        highlights = (gray_frame >= 192)
        
        violations = mask > 0
        
        shadow_violations = np.sum(violations & shadows)
        midtone_violations = np.sum(violations & midtones)
        highlight_violations = np.sum(violations & highlights)
        
        total_violations = max(1, shadow_violations + midtone_violations + highlight_violations)
        
        return {
            'shadow_ratio': float(shadow_violations / total_violations),
            'midtone_ratio': float(midtone_violations / total_violations),
            'highlight_ratio': float(highlight_violations / total_violations),
            'primary_zone': self._get_primary_zone(shadow_violations, midtone_violations, highlight_violations)
        }
    
    def _generate_diagnostic(self, spatial_patterns, systematic_patterns, luma_distribution, boundary_artifacts=None):
        """Generate human-readable diagnostic based on pattern analysis"""
        diagnostics = []
        
        # Check for boundary artifacts FIRST (highest priority)
        if boundary_artifacts:
            edges_detected = []
            if boundary_artifacts.get('left_edge'):
                edges_detected.append('left')
            if boundary_artifacts.get('right_edge'):
                edges_detected.append('right')
            if boundary_artifacts.get('top_edge'):
                edges_detected.append('top')
            if boundary_artifacts.get('bottom_edge'):
                edges_detected.append('bottom')
            
            if edges_detected:
                edge_str = '/'.join(edges_detected)
                diagnostics.append(f"Border/blanking artifacts ({edge_str} edge)")
        
        # Spatial pattern diagnostics (skip if boundary artifacts detected)
        if not (boundary_artifacts and any([boundary_artifacts.get('left_edge'), 
                                            boundary_artifacts.get('right_edge'),
                                            boundary_artifacts.get('top_edge'),
                                            boundary_artifacts.get('bottom_edge')])):
            if spatial_patterns.get('edge_concentrated'):
                diagnostics.append("Edge enhancement artifacts")
            if spatial_patterns.get('perimeter_concentrated'):
                diagnostics.append("Perimeter artifacts (possible scaling issues)")
            if spatial_patterns.get('scattered'):
                diagnostics.append("Scattered noise-like violations")
        
        # Systematic pattern diagnostics
        if systematic_patterns.get('horizontal_banding'):
            diagnostics.append("Horizontal banding (interlacing artifacts)")
        if systematic_patterns.get('vertical_banding'):
            # Only report vertical banding if it's not a boundary artifact
            if not (boundary_artifacts and (boundary_artifacts.get('left_edge') or boundary_artifacts.get('right_edge'))):
                diagnostics.append("Vertical banding")
        if systematic_patterns.get('macroblock_artifacts'):
            diagnostics.append("Macroblock-aligned artifacts")
        
        # Luma zone diagnostics (only if not boundary artifacts)
        # Boundary artifacts often appear dark but aren't actually shadow crushing
        if not (boundary_artifacts and any([boundary_artifacts.get('left_edge'), 
                                            boundary_artifacts.get('right_edge'),
                                            boundary_artifacts.get('top_edge'),
                                            boundary_artifacts.get('bottom_edge')])):
            primary_zone = luma_distribution.get('primary_zone')
            if primary_zone == 'highlights' and luma_distribution.get('highlight_ratio', 0) > 0.7:
                diagnostics.append("Highlight clipping")
            elif primary_zone == 'shadows' and luma_distribution.get('shadow_ratio', 0) > 0.7:
                # Double-check this isn't a misidentified border
                if not spatial_patterns.get('left_edge_line') and not spatial_patterns.get('right_edge_line'):
                    diagnostics.append("Shadow crushing")
        
        return diagnostics if diagnostics else ["General broadcast range violations"]
    
    def analyze_temporal_consistency(self, frame_violations_timeline):
        """
        Analyze how violations change over time to identify persistent vs transient issues.
        """
        if len(frame_violations_timeline) < 2:
            return None
        
        # Convert to numpy array for analysis
        timeline = np.array([(f['timestamp'], f['violation_percentage']) 
                            for f in frame_violations_timeline])
        
        timestamps = timeline[:, 0]
        violations = timeline[:, 1]
        
        # Calculate temporal metrics
        results = {
            'mean_violation': float(np.mean(violations)),
            'std_violation': float(np.std(violations)),
            'max_violation': float(np.max(violations)),
            'min_violation': float(np.min(violations)),
            
            # Persistence analysis
            'persistent_threshold': float(np.percentile(violations, 25)),
            'frames_above_threshold': int(np.sum(violations > np.percentile(violations, 25))),
            
            # Trend analysis
            'trend': self._calculate_trend(timestamps, violations),
            
            # Burst detection
            'violation_bursts': self._detect_bursts(timestamps, violations),
            
            # Consistency score (0-1, higher = more consistent)
            'consistency_score': float(1.0 - (np.std(violations) / (np.mean(violations) + 0.001)))
        }
        
        # Classify temporal pattern
        if results['consistency_score'] > 0.8:
            results['temporal_pattern'] = 'persistent'
            results['interpretation'] = 'Consistent violations suggest systematic level issues'
        elif len(results['violation_bursts']) > 5:
            results['temporal_pattern'] = 'intermittent'
            results['interpretation'] = 'Intermittent bursts suggest scene-specific issues'
        elif results['trend'] > 0.1:
            results['temporal_pattern'] = 'increasing'
            results['interpretation'] = 'Increasing violations may indicate progressive degradation'
        elif results['trend'] < -0.1:
            results['temporal_pattern'] = 'decreasing'
            results['interpretation'] = 'Decreasing violations suggest improvement over time'
        else:
            results['temporal_pattern'] = 'sporadic'
            results['interpretation'] = 'Sporadic violations suggest isolated problem frames'
        
        return results
    
    def _calculate_trend(self, timestamps, violations):
        """Calculate trend in violations over time"""
        if len(timestamps) < 2:
            return 0.0
        
        # Linear regression
        coeffs = np.polyfit(timestamps, violations, 1)
        return float(coeffs[0])
    
    def _detect_bursts(self, timestamps, violations, threshold_percentile=75):
        """Detect bursts of high violations"""
        threshold = np.percentile(violations, threshold_percentile)
        bursts = []
        in_burst = False
        burst_start = None
        
        for i, (t, v) in enumerate(zip(timestamps, violations)):
            if v > threshold and not in_burst:
                in_burst = True
                burst_start = i
            elif v <= threshold and in_burst:
                in_burst = False
                if burst_start is not None:
                    bursts.append({
                        'start_time': float(timestamps[burst_start]),
                        'end_time': float(timestamps[i-1]),
                        'duration': float(timestamps[i-1] - timestamps[burst_start]),
                        'max_violation': float(np.max(violations[burst_start:i]))
                    })
        
        # Handle burst that extends to end
        if in_burst and burst_start is not None:
            bursts.append({
                'start_time': float(timestamps[burst_start]),
                'end_time': float(timestamps[-1]),
                'duration': float(timestamps[-1] - timestamps[burst_start]),
                'max_violation': float(np.max(violations[burst_start:]))
            })
        
        return bursts
    
    def generate_actionable_report(self, analysis_results):
        """
        Generate specific, actionable recommendations based on violation patterns.
        """
        recommendations = []
        severity_score = 0
        
        # Analyze patterns across all samples
        primary_zone = analysis_results.get('aggregate_patterns', {}).get('primary_violation_zone')
        edge_ratio = analysis_results.get('aggregate_patterns', {}).get('avg_edge_violation_ratio', 0)
        temporal = analysis_results.get('temporal_analysis', {})
        
        # Check for boundary artifacts first (these are usually less severe)
        boundary_edges = analysis_results.get('aggregate_patterns', {}).get('boundary_edges_detected', [])
        if boundary_edges:
            edge_list = list(set(boundary_edges))  # Unique edges
            recommendations.append({
                'issue': 'Border/Blanking Artifacts',
                'severity': 'low',
                'affected_areas': f"Frame edges: {', '.join(edge_list)}",
                'description': 'Residual blanking or border areas not fully cropped',
                'percentage_affected': analysis_results.get('aggregate_patterns', {}).get('boundary_artifact_percentage', 0)
            })
            severity_score += 1
        
        # Only report shadow/highlight issues if not boundary artifacts
        if not boundary_edges:
            if primary_zone == 'highlights':
                recommendations.append({
                    'issue': 'Highlight Clipping',
                    'severity': 'high',
                    'affected_areas': 'Bright areas of the image',
                    'percentage_affected': analysis_results.get('aggregate_patterns', {}).get('highlight_percentage', 0)
                })
                severity_score += 3
            
            if primary_zone == 'shadows':
                recommendations.append({
                    'issue': 'Shadow Crushing',
                    'severity': 'medium',
                    'affected_areas': 'Dark areas of the image',
                    'percentage_affected': analysis_results.get('aggregate_patterns', {}).get('shadow_percentage', 0)
                })
                severity_score += 2
        
        if edge_ratio > 0.5 and not boundary_edges:
            recommendations.append({
                'issue': 'Edge Enhancement Artifacts',
                'severity': 'medium',
                'affected_areas': 'Object edges and fine details',
                'percentage_affected': edge_ratio * 100
            })
            severity_score += 2
        
        if analysis_results.get('aggregate_patterns', {}).get('horizontal_banding_frames', 0) > 0:
            recommendations.append({
                'issue': 'Interlacing/Scan Line Artifacts',
                'severity': 'high',
                'affected_areas': 'Horizontal bands across frame',
                'frames_affected': analysis_results['aggregate_patterns']['horizontal_banding_frames']
            })
            severity_score += 3
        
        if analysis_results.get('aggregate_patterns', {}).get('macroblock_frames', 0) > 0:
            recommendations.append({
                'issue': 'Compression Block Artifacts',
                'severity': 'medium',
                'affected_areas': 'Block boundaries in image',
                'frames_affected': analysis_results['aggregate_patterns']['macroblock_frames']
            })
            severity_score += 2
        
        # Temporal pattern-based recommendations
        if temporal and temporal.get('temporal_pattern') == 'persistent':
            severity_score += 1
            for rec in recommendations:
                rec['temporal_nature'] = 'Persistent throughout video'
        elif temporal and temporal.get('temporal_pattern') == 'intermittent':
            for rec in recommendations:
                rec['temporal_nature'] = 'Occurs in specific scenes'
        
        # Generate overall assessment
        if severity_score == 0:
            overall_assessment = "Video is broadcast-safe with no significant issues"
            action_priority = "none"
        elif severity_score < 3:
            overall_assessment = "Minor broadcast range issues detected"
            action_priority = "low"
        elif severity_score < 6:
            overall_assessment = "Moderate broadcast range issues requiring attention"
            action_priority = "medium"
        else:
            overall_assessment = "Significant broadcast range violations requiring correction"
            action_priority = "high"
        
        # Get problem timestamp ranges
        timestamp_ranges = self._get_problem_timestamp_ranges(analysis_results)
        
        return {
            'overall_assessment': overall_assessment,
            'action_priority': action_priority,
            'severity_score': severity_score,
            'recommendations': recommendations,
            'timestamp_ranges': timestamp_ranges,
            'summary_statistics': {
                'total_frames_analyzed': analysis_results.get('total_frames_analyzed', 0),
                'frames_with_violations': analysis_results.get('frames_with_violations', 0),
                'average_violation_percentage': analysis_results.get('average_violation_percentage', 0),
                'max_violation_percentage': analysis_results.get('max_violation_percentage', 0)
            }
        }
    
    def _get_problem_timestamp_ranges(self, analysis_results):
        """Extract timestamp ranges where problems occur"""
        ranges = []
        
        temporal = analysis_results.get('temporal_analysis', {})
        if temporal and 'violation_bursts' in temporal:
            for burst in temporal['violation_bursts']:
                ranges.append({
                    'start': self.format_timecode(burst['start_time']),
                    'end': self.format_timecode(burst['end_time']),
                    'severity': 'high' if burst['max_violation'] > 1.0 else 'medium'
                })
        
        return ranges
    
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
        print(f"\nSaving diagnostic thumbnails for {num_thumbnails} worst frames...")
        
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
                
                # Bottom-left: Violation mask
                violation_mask = self.detect_brng_violations_differential(frame_h, frame_o)
                violation_colored = cv2.cvtColor(violation_mask, cv2.COLOR_GRAY2BGR)
                violation_colored[:, :, 1] = violation_mask  # Make it greenish
                viz[h:, :w] = violation_colored
                
                # Bottom-right: Analysis overlay on original
                overlay = frame_o.copy()
                violation_overlay = np.zeros_like(overlay)
                violation_overlay[:, :, 2] = violation_mask  # Red channel
                overlay = cv2.addWeighted(overlay, 0.7, violation_overlay, 0.3, 0)
                viz[h:, w:] = overlay
                
                # Add text labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(viz, "Original", (10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(viz, "BRNG Highlighted", (w+10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(viz, "Violations Only", (10, h+30), font, 1, (255, 255, 255), 2)
                cv2.putText(viz, "Overlay", (w+10, h+30), font, 1, (255, 255, 255), 2)
                
                # Add analysis info
                info_text = [
                    f"Time: {self.format_timecode(timestamp)}",
                    f"Violations: {frame_data['violation_percentage']:.4f}%",
                    f"Pattern: {', '.join(frame_data.get('diagnostics', ['Unknown']))}"
                ]
                
                y_offset = h*2 - 80
                for line in info_text:
                    cv2.putText(viz, line, (10, y_offset), font, 0.6, (255, 255, 255), 1)
                    y_offset += 25
                
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
                
                print(f"  ✔ Saved: {thumbnail_filename}")
        
        return saved_thumbnails
    
    def analyze_video_comprehensive(self, duration_limit=300):
        """
        Comprehensive analysis using differential detection and adaptive sampling
        """
        print(f"\nPerforming comprehensive BRNG analysis...")
        
        # Open both videos
        cap_highlighted = cv2.VideoCapture(str(self.highlighted_video))
        cap_original = cv2.VideoCapture(str(self.original_video))
        
        if not cap_highlighted.isOpened() or not cap_original.isOpened():
            print("✗ Could not open processed videos")
            return None
        
        # Get video properties from highlighted version
        fps = cap_highlighted.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_highlighted.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap_highlighted.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_highlighted.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
        
        # Adaptive sampling
        frame_samples = self.adaptive_video_sampling(cap_highlighted, cap_original)
        
        # Analyze samples
        frame_violations = []
        aggregate_patterns = defaultdict(list)
        boundary_edges_detected = []
        
        print(f"\nAnalyzing {len(frame_samples)} samples...")
        
        for idx, (frame_idx, frame_h, frame_o) in enumerate(frame_samples):
            # Differential detection
            violation_mask = self.detect_brng_violations_differential(frame_h, frame_o)
            violation_pixels = int(np.sum(violation_mask > 0))
            
            if violation_pixels > 0:
                # Pattern analysis
                pattern_analysis = self.analyze_violation_patterns(violation_mask, frame_o)
                
                # Store results
                timestamp = frame_idx / fps
                frame_violations.append({
                    'frame': int(frame_idx),
                    'timestamp': float(timestamp),
                    'timecode': self.format_timecode(timestamp),
                    'violation_pixels': int(violation_pixels),
                    'violation_percentage': float((violation_pixels / (width * height)) * 100),
                    'pattern_analysis': pattern_analysis,
                    'diagnostics': pattern_analysis['diagnostic']
                })
                
                # Aggregate patterns
                aggregate_patterns['edge_ratios'].append(pattern_analysis['edge_violation_ratio'])
                
                # Track boundary artifacts
                if pattern_analysis.get('boundary_artifacts'):
                    ba = pattern_analysis['boundary_artifacts']
                    if ba.get('left_edge'):
                        boundary_edges_detected.append('left')
                    if ba.get('right_edge'):
                        boundary_edges_detected.append('right')
                    if ba.get('top_edge'):
                        boundary_edges_detected.append('top')
                    if ba.get('bottom_edge'):
                        boundary_edges_detected.append('bottom')
                
                # Only count luma zones if not boundary artifacts
                if not pattern_analysis.get('boundary_artifacts', {}).get('left_edge') and \
                   not pattern_analysis.get('boundary_artifacts', {}).get('right_edge'):
                    aggregate_patterns['primary_zones'].append(pattern_analysis['luma_distribution']['primary_zone'])
                
                if pattern_analysis['systematic_patterns']['horizontal_banding']:
                    aggregate_patterns['horizontal_banding_frames'].append(frame_idx)
                if pattern_analysis['systematic_patterns']['macroblock_artifacts']:
                    aggregate_patterns['macroblock_frames'].append(frame_idx)
            
            # Progress indicator
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(frame_samples)} samples")
        
        # Calculate aggregate statistics
        if frame_violations:
            avg_violation_pct = np.mean([f['violation_percentage'] for f in frame_violations])
            max_violation_pct = np.max([f['violation_percentage'] for f in frame_violations])
            
            # Determine primary patterns
            if aggregate_patterns['primary_zones']:
                zone_counts = defaultdict(int)
                for zone in aggregate_patterns['primary_zones']:
                    zone_counts[zone] += 1
                primary_zone = max(zone_counts, key=zone_counts.get)
            else:
                primary_zone = 'none'
            
            # Calculate boundary artifact statistics
            boundary_artifact_frames = len([f for f in frame_violations 
                                           if any(f.get('pattern_analysis', {}).get('boundary_artifacts', {}).values())])
            boundary_artifact_percentage = (boundary_artifact_frames / len(frame_violations)) * 100 if frame_violations else 0
            
            aggregate_summary = {
                'primary_violation_zone': primary_zone,
                'avg_edge_violation_ratio': float(np.mean(aggregate_patterns['edge_ratios'])) if aggregate_patterns['edge_ratios'] else 0.0,
                'horizontal_banding_frames': len(aggregate_patterns.get('horizontal_banding_frames', [])),
                'macroblock_frames': len(aggregate_patterns.get('macroblock_frames', [])),
                'highlight_percentage': sum(1 for z in aggregate_patterns['primary_zones'] if z == 'highlights') / len(aggregate_patterns['primary_zones']) * 100 if aggregate_patterns['primary_zones'] else 0,
                'shadow_percentage': sum(1 for z in aggregate_patterns['primary_zones'] if z == 'shadows') / len(aggregate_patterns['primary_zones']) * 100 if aggregate_patterns['primary_zones'] else 0,
                'boundary_edges_detected': boundary_edges_detected,
                'boundary_artifact_frames': boundary_artifact_frames,
                'boundary_artifact_percentage': boundary_artifact_percentage
            }
        else:
            avg_violation_pct = 0.0
            max_violation_pct = 0.0
            aggregate_summary = {}
        
        # Temporal analysis
        temporal_analysis = self.analyze_temporal_consistency(frame_violations) if frame_violations else None
        
        # Find worst frames
        worst_frames = sorted(frame_violations, 
                            key=lambda x: x['violation_pixels'], 
                            reverse=True)[:10]
        
        # Save diagnostic thumbnails
        saved_thumbnails = []
        if worst_frames:
            saved_thumbnails = self.save_diagnostic_thumbnails(
                worst_frames, cap_highlighted, cap_original, num_thumbnails=5
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
                'active_area': list(self.active_area) if self.active_area else None
            },
            'analysis_method': 'differential_detection',
            'total_frames_analyzed': len(frame_samples),
            'frames_with_violations': len(frame_violations),
            'average_violation_percentage': float(avg_violation_pct),
            'max_violation_percentage': float(max_violation_pct),
            'aggregate_patterns': aggregate_summary,
            'temporal_analysis': temporal_analysis,
            'worst_frames': worst_frames[:5],  # Top 5 worst
            'saved_thumbnails': saved_thumbnails
        }
        
        # Generate actionable report
        actionable_report = self.generate_actionable_report(analysis)
        analysis['actionable_report'] = actionable_report
        
        # Before saving, convert all numpy types
        analysis = self._convert_numpy_types(analysis)

        # Save analysis
        with open(self.analysis_output, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✔ Analysis complete. Results saved to: {self.analysis_output}")
        if saved_thumbnails:
            print(f"✔ Saved {len(saved_thumbnails)} diagnostic thumbnail(s) to: {self.thumbnails_dir}")
        
        return analysis
    
    def print_summary(self, analysis):
        """Print enhanced analysis summary"""
        print("\n" + "="*80)
        print("ACTIVE AREA BRNG ANALYSIS RESULTS - ENHANCED")
        print("="*80)
        
        info = analysis['video_info']
        
        if info['active_area']:
            print(f"Active Area: {info['active_area'][2]}x{info['active_area'][3]} at ({info['active_area'][0]},{info['active_area'][1]})")
        else:
            print("Analyzed: Full frame (no border data)")
        
        print(f"Video: {info['width']}x{info['height']}, {info['duration']:.1f}s")
        print(f"Frames analyzed: {analysis['total_frames_analyzed']}")
        print(f"Analysis method: Differential detection (eliminates false positives)")
        
        report = analysis.get('actionable_report', {})
        
        print(f"\n{report.get('overall_assessment', 'Analysis complete')}")
        print(f"Priority: {report.get('action_priority', 'none').upper()}")
        
        if analysis['frames_with_violations'] > 0:
            print(f"\nVIOLATION STATISTICS:")
            print(f"  Frames with violations: {analysis['frames_with_violations']}/{analysis['total_frames_analyzed']}")
            print(f"  Average violation: {analysis['average_violation_percentage']:.4f}% of pixels")
            print(f"  Maximum violation: {analysis['max_violation_percentage']:.4f}% of pixels")
        
        # Temporal analysis
        temporal = analysis.get('temporal_analysis')
        if temporal:
            print(f"\nTEMPORAL PATTERN: {temporal.get('temporal_pattern', 'unknown')}")
            print(f"  {temporal.get('interpretation', '')}")
            if temporal.get('violation_bursts'):
                print(f"  Detected {len(temporal['violation_bursts'])} violation burst(s)")
        
        # Specific issues
        if report.get('recommendations'):
            print("\nSPECIFIC ISSUES DETECTED:")
            for rec in report['recommendations']:
                print(f"  • {rec['issue']} ({rec['severity']} severity)")
                print(f"    Affects: {rec['affected_areas']}")
                if 'description' in rec:
                    print(f"    Description: {rec['description']}")
                if 'temporal_nature' in rec:
                    print(f"    Nature: {rec['temporal_nature']}")
        
        # Check for boundary artifacts specifically
        aggregate = analysis.get('aggregate_patterns', {})
        if aggregate.get('boundary_artifact_frames', 0) > 0:
            print(f"\n⚠️ BOUNDARY ARTIFACT NOTE:")
            print(f"  {aggregate['boundary_artifact_frames']} frames show edge artifacts")
            print(f"  This suggests the active area detection may have missed some blanking")
            print(f"  Consider re-running border detection with adjusted parameters")
        
        # Problem timestamps
        if report.get('timestamp_ranges'):
            print("\nPROBLEM TIMESTAMP RANGES:")
            for range_info in report['timestamp_ranges'][:5]:  # Show first 5
                print(f"  {range_info['start']} - {range_info['end']} ({range_info['severity']})")
        
        # Worst frames
        if analysis.get('worst_frames'):
            print("\nWORST FRAMES:")
            for i, frame in enumerate(analysis['worst_frames'][:3], 1):
                print(f"  {i}. Frame {frame['frame']} ({frame['timecode']}) - {frame['violation_percentage']:.4f}% pixels")
                if frame.get('diagnostics'):
                    print(f"     Issues: {', '.join(frame['diagnostics'][:2])}")
        
        print("="*80)
    
    def cleanup(self, keep_intermediates=False):
        """
        Clean up temporary files
        """
        try:
            if not keep_intermediates:
                if self.highlighted_video.exists():
                    self.highlighted_video.unlink()
                if self.original_video.exists():
                    self.original_video.unlink()
                print("✔ Removed intermediate videos")
            else:
                print(f"ℹ️ Kept intermediate videos in: {self.temp_dir}")
            
            # Remove temp directory if empty
            if self.temp_dir.exists():
                if not any(self.temp_dir.iterdir()):
                    self.temp_dir.rmdir()
                    print("✔ Cleaned up temporary directory")
        except Exception as e:
            print(f"⚠️ Could not clean up temp files: {e}")


def analyze_active_area_brng(video_path, border_data_path=None, output_dir=None, 
                            duration_limit=300):
    """
    Main function to analyze BRNG violations with enhanced differential detection.
    
    Args:
        video_path: Path to video file
        border_data_path: Path to border detection JSON
        output_dir: Output directory for results
        duration_limit: Maximum seconds to process
    
    Returns:
        Analysis results dictionary
    """
    analyzer = ActiveAreaBrngAnalyzer(video_path, border_data_path, output_dir)
    
    # Process with ffmpeg to create both highlighted and original versions
    if not analyzer.process_with_ffmpeg(duration_limit):
        print("FFmpeg processing failed")
        return None
    
    # Comprehensive analysis with differential detection
    analysis = analyzer.analyze_video_comprehensive(duration_limit)
    
    if analysis:
        analyzer.print_summary(analysis)
    
    # Clean up temp files
    analyzer.cleanup(keep_intermediates=False)
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze BRNG violations with differential detection and enhanced pattern recognition'
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