#!/usr/bin/env python3
"""
Video Border Detection Script

Detects blanking borders and active picture areas in video files using OpenCV.
This script focuses specifically on border detection and can be extended
with additional border analysis functionality.
"""

import cv2
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


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
        
        print(f"✓ Video loaded: {self.width}x{self.height}, {self.fps:.2f}fps, {self.duration:.1f}s")
        
    def detect_blanking_borders(self, threshold=10, edge_sample_width=100, avoid_dark_frames=True):
        """
        Detect borders with improved accuracy, especially for right side
        
        Args:
            threshold: Brightness threshold for border detection
            edge_sample_width: Width of edge sampling area
            avoid_dark_frames: If True, skip very dark frames that might be fades/transitions
        """
        frame_indices = np.linspace(0, self.total_frames - 1, 
                                   min(self.sample_frames, self.total_frames), 
                                   dtype=int)
        
        left_borders = []
        right_borders = []
        top_borders = []
        bottom_borders = []
        frames_used = 0
        frames_skipped = 0
        
        print(f"Analyzing up to {len(frame_indices)} frames for border detection...")
        
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Skip very dark frames if requested (likely fades, black frames, etc.)
            if avoid_dark_frames:
                mean_brightness = np.mean(gray)
                if mean_brightness < 20:  # Very dark frame
                    frames_skipped += 1
                    continue
            
            frames_used += 1
            
            # Detect left border - scan from left
            left = 0
            for x in range(min(edge_sample_width, w)):
                # Use mean instead of max for more robust detection
                if np.mean(gray[:, x]) > threshold:
                    left = x
                    break
                    
            # Detect right border - scan from right, more thorough
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
        
        if frames_skipped > 0:
            print(f"  Skipped {frames_skipped} dark frames, used {frames_used} frames for detection")
        else:
            print(f"  Used {frames_used} frames for detection")
            
        if not left_borders:
            return None
            
        # Calculate stable borders with more conservative approach
        median_left = int(np.median(left_borders))
        median_right = int(np.median(right_borders))
        
        # Add some padding for tighter active area
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
            print("Warning: Detected active area seems too small")
            return None
            
        result = (median_left, mode_top, active_width, active_height)
        
        print(f"\nBorder detection statistics:")
        print(f"  Left border: median={median_left} (std={np.std(left_borders):.1f})")
        print(f"  Right border: median={median_right} (std={np.std(right_borders):.1f})")
        print(f"  Top border: mode={mode_top} (variations={len(top_unique)})")
        print(f"  Bottom border: mode={mode_bottom} (variations={len(bottom_unique)})")
        print(f"  Active area (with {padding}px padding): {active_width}x{active_height} at ({median_left},{mode_top})")
        
        return result
    
    def detect_head_switching_artifacts(self, active_area=None, sample_frames=20):
        """
        Detect head switching artifacts at the bottom of the picture area.
        
        Head switching artifacts typically manifest as:
        - Bottom horizontal line(s) only occupying left half of screen
        - Bottom right corner appearing black/missing
        - Horizontal discontinuity at the video head switching point
        
        Args:
            active_area: tuple (x, y, w, h) of active picture area, or None for full frame
            sample_frames: number of frames to analyze for consistency
            
        Returns:
            Dictionary with artifact detection results
        """
        try:
            if active_area:
                crop_x, crop_y, crop_w, crop_h = active_area
                # Analyze bottom of active area
                analysis_y = crop_y + crop_h - 10  # Bottom 10 lines of active area
                analysis_height = 10
            else:
                crop_x, crop_y, crop_w, crop_h = 0, 0, self.width, self.height
                # Analyze bottom of full frame
                analysis_y = self.height - 15  # Bottom 15 lines
                analysis_height = 15
            
            # Ensure we don't go out of bounds
            analysis_y = max(0, analysis_y)
            analysis_height = min(analysis_height, self.height - analysis_y)
            
            if analysis_height <= 0 or crop_w <= 0:
                print("⚠️ Invalid analysis region for head switching detection")
                return {
                    'frames_analyzed': 0,
                    'frames_with_artifacts': 0,
                    'artifact_percentage': 0.0,
                    'severity': 'none',
                    'error': 'Invalid analysis region'
                }
            
            print(f"\nAnalyzing bottom {analysis_height} lines for head switching artifacts...")
            print(f"Analysis region: {crop_w}x{analysis_height} at ({crop_x},{analysis_y})")
            
            # Sample frames for analysis
            frame_indices = np.linspace(0, self.total_frames - 1, 
                                       min(sample_frames, self.total_frames), 
                                       dtype=int)
            
            artifact_detections = []
            line_asymmetry_scores = []
            horizontal_discontinuities = []
            
            for idx in frame_indices:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    continue
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Extract the bottom region for analysis
                if analysis_y + analysis_height > gray.shape[0] or crop_x + crop_w > gray.shape[1]:
                    continue  # Skip if region is out of bounds
                    
                bottom_region = gray[analysis_y:analysis_y + analysis_height, crop_x:crop_x + crop_w]
                
                if bottom_region.size == 0:
                    continue
                
                # Analyze each line in the bottom region
                frame_asymmetries = []
                frame_discontinuities = []
                
                for line_idx in range(bottom_region.shape[0]):
                    line = bottom_region[line_idx, :]
                    line_width = len(line)
                    
                    if line_width < 20:  # Skip very narrow lines
                        continue
                    
                    # Split line into left and right halves
                    mid_point = line_width // 2
                    left_half = line[:mid_point]
                    right_half = line[mid_point:]
                    
                    # Calculate brightness for each half
                    left_brightness = np.mean(left_half)
                    right_brightness = np.mean(right_half)
                    
                    # Calculate asymmetry score (higher = more asymmetric)
                    if left_brightness > 10:  # Avoid division by very small numbers
                        asymmetry = abs(left_brightness - right_brightness) / left_brightness
                    else:
                        asymmetry = 0
                    
                    frame_asymmetries.append(asymmetry)
                    
                    # Look for horizontal discontinuities (abrupt changes)
                    # Check for sudden drops in brightness from left to right
                    if left_brightness > 30 and right_brightness < 15:
                        frame_discontinuities.append(line_idx)
                    
                    # Also check for sharp transitions within the line
                    if line_width > 10:
                        # Look for sharp drops in the rightmost quarter
                        right_quarter = line[int(line_width * 0.75):]
                        left_three_quarters = line[:int(line_width * 0.75)]
                        
                        if len(right_quarter) > 0 and len(left_three_quarters) > 0:
                            if np.mean(left_three_quarters) > 30 and np.mean(right_quarter) < 15:
                                frame_discontinuities.append(line_idx)
                
                # Store results for this frame
                if frame_asymmetries:
                    avg_asymmetry = np.mean(frame_asymmetries)
                    max_asymmetry = np.max(frame_asymmetries)
                    line_asymmetry_scores.append(avg_asymmetry)
                    
                    # Consider it an artifact if multiple lines show high asymmetry
                    artifact_lines = sum(1 for asym in frame_asymmetries if asym > 0.5)
                    if artifact_lines >= 2:  # At least 2 lines with significant asymmetry
                        artifact_detections.append({
                            'frame_idx': int(idx),
                            'time': float(idx / self.fps),
                            'avg_asymmetry': float(avg_asymmetry),
                            'max_asymmetry': float(max_asymmetry),
                            'artifact_lines': int(artifact_lines),
                            'discontinuities': len(frame_discontinuities)
                        })
                
                horizontal_discontinuities.extend(frame_discontinuities)
            
            # Analyze results
            results = {
                'frames_analyzed': len(frame_indices),
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
                'artifact_frames': artifact_detections[:5],  # Store first 5 examples
                'severity': 'none'
            }
            
            if line_asymmetry_scores:
                results['avg_asymmetry'] = float(np.mean(line_asymmetry_scores))
                results['max_asymmetry'] = float(np.max(line_asymmetry_scores))
            
            if len(frame_indices) > 0:
                results['artifact_percentage'] = float((len(artifact_detections) / len(frame_indices)) * 100)
            
            # Determine severity
            if results['artifact_percentage'] > 50:
                results['severity'] = 'severe'
            elif results['artifact_percentage'] > 20:
                results['severity'] = 'moderate'  
            elif results['artifact_percentage'] > 5:
                results['severity'] = 'minor'
            
            # Report findings
            print(f"  Frames with head switching artifacts: {results['frames_with_artifacts']}/{results['frames_analyzed']} ({results['artifact_percentage']:.1f}%)")
            print(f"  Average asymmetry score: {results['avg_asymmetry']:.3f}")
            print(f"  Horizontal discontinuities found: {results['total_discontinuities']}")
            print(f"  Severity: {results['severity']}")
            
            if results['severity'] != 'none':
                print(f"  ⚠️  Head switching artifacts detected - check bottom of picture area")
            else:
                print(f"  ✓ No significant head switching artifacts detected")
            
            return results
            
        except Exception as e:
            print(f"⚠️ Error in head switching analysis: {e}")
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
            print("No active area detected, cannot analyze borders")
            return None
            
        x, y, w, h = active_area
        regions = {}
        
        # Left border
        if x > 10:
            regions['left_border'] = (0, 0, int(x), int(self.height))
            print(f"Left border region: {x}px wide")
        
        # Right border  
        right_border_start = x + w
        if right_border_start < self.width - 10:
            right_width = self.width - right_border_start
            regions['right_border'] = (int(right_border_start), 0, int(right_width), int(self.height))
            print(f"Right border region: {right_width}px wide")
        
        # Top border
        if y > 10:
            regions['top_border'] = (0, 0, int(self.width), int(y))
            print(f"Top border region: {y}px tall")
        
        # Bottom border
        bottom_border_start = y + h
        if bottom_border_start < self.height - 10:
            bottom_height = self.height - bottom_border_start
            regions['bottom_border'] = (0, int(bottom_border_start), int(self.width), int(bottom_height))
            print(f"Bottom border region: {bottom_height}px tall")
            
        return regions
    
    def find_good_representative_frame(self, target_time=150, search_window=120):
        """
        Find a good representative frame with sufficient brightness and detail
        
        Args:
            target_time: Preferred time in seconds
            search_window: Seconds to search around target time (default: 2 minutes)
        
        Returns:
            Best frame found, or None if no suitable frame
        """
        # Calculate search range
        target_frame = int(target_time * self.fps)
        window_frames = int(search_window * self.fps)
        
        start_frame = max(0, target_frame - window_frames // 2)
        end_frame = min(self.total_frames - 1, target_frame + window_frames // 2)
        
        # If video is too short, search the middle section
        if end_frame >= self.total_frames:
            mid_point = self.total_frames // 2
            start_frame = max(0, mid_point - window_frames // 2)
            end_frame = min(self.total_frames - 1, mid_point + window_frames // 2)
        
        print(f"Searching for good representative frame between {start_frame/self.fps:.1f}s and {end_frame/self.fps:.1f}s...")
        
        best_frame = None
        best_score = 0
        best_frame_idx = target_frame
        candidates_found = 0
        
        # Check frames every 1 second in the search window
        check_interval = max(1, int(self.fps))
        
        # First pass: Look for ideal frames
        for frame_idx in range(start_frame, end_frame, check_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                continue
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame quality metrics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # First pass: stricter criteria
            # Avoid very dark frames (mean < 25) and very bright frames (mean > 230)  
            if mean_brightness < 25 or mean_brightness > 230:
                continue
                
            # Must have reasonable contrast
            if std_brightness < 15:
                continue
                
            candidates_found += 1
                
            # Score based on brightness and contrast
            brightness_score = 1.0 - abs(mean_brightness - 120) / 120
            contrast_score = min(std_brightness / 40.0, 1.0)
            
            # Combined score favoring contrast over brightness
            score = brightness_score * 0.3 + contrast_score * 0.7
            
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_frame_idx = frame_idx
        
        # Second pass: relaxed criteria if no good frame found
        if best_frame is None:
            print(f"No ideal frames found, searching with relaxed criteria...")
            
            for frame_idx in range(start_frame, end_frame, check_interval):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                std_brightness = np.std(gray)
                
                # Relaxed criteria: just avoid completely black/white frames
                if mean_brightness < 15 or mean_brightness > 245:
                    continue
                    
                candidates_found += 1
                    
                # Simple scoring for any decent frame
                score = mean_brightness / 255.0 + std_brightness / 100.0
                
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
                    best_frame_idx = frame_idx
        
        # Third pass: if still no frame, just find anything that's not completely black
        if best_frame is None:
            print(f"Still no suitable frame, looking for any non-black frame...")
            
            for frame_idx in range(start_frame, end_frame, check_interval * 2):  # Check less frequently
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                
                # Just avoid completely black frames
                if mean_brightness > 10:
                    best_frame = frame.copy()
                    best_frame_idx = frame_idx
                    candidates_found += 1
                    break
        
        if best_frame is not None:
            print(f"✓ Selected frame at {best_frame_idx/self.fps:.1f}s (score: {best_score:.2f}, candidates: {candidates_found})")
            return best_frame
        else:
            print(f"⚠️ No suitable frame found in {search_window}s window, using target frame")
            # Fallback to original behavior
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame if target_frame < self.total_frames else self.total_frames // 2)
            ret, frame = self.cap.read()
            return frame if ret else None

    def generate_border_visualization(self, output_path, active_area=None, head_switching_results=None, target_time=150, search_window=120):
        """
        Generate visual showing detected borders and active area
        
        Args:
            target_time: Target time for frame selection (seconds)
            search_window: How many seconds to search around target time
            head_switching_results: Results from head switching analysis to highlight regions
        """
        # Find a good representative frame instead of using fixed time
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
                if hs_region:
                    hs_x = hs_region.get('x', 0)
                    hs_y = hs_region.get('y', self.height - 10)
                    hs_w = hs_region.get('width', self.width)
                    
                    # Draw a horizontal line at the bottom of the analysis region
                    line_y = hs_y + hs_region.get('height', 10) - 1  # Bottom of analysis region
                    ax1.plot([hs_x, hs_x + hs_w], [line_y, line_y], 
                            color='orange', linewidth=2, alpha=0.8, 
                            label='Head Switching Artifacts')
            
            # Only show legend if we have patches to show
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
            
            # Add head switching info if relevant
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
                # Highlight head switching analysis region on full frame
                hs_region = head_switching_results.get('analysis_region', {})
                if hs_region:
                    hs_x = hs_region.get('x', 0)
                    hs_y = hs_region.get('y', self.height - 15)
                    hs_w = hs_region.get('width', self.width)
                    
                    # Draw a horizontal line at the bottom of the analysis region
                    line_y = hs_y + hs_region.get('height', 15) - 1  # Bottom of analysis region
                    ax1.plot([hs_x, hs_x + hs_w], [line_y, line_y], 
                            color='orange', linewidth=2, alpha=0.8, 
                            label='Head Switching Artifacts')
                    ax1.legend()
                
                # Add head switching info
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
        Save border detection results to JSON file
        """
        border_regions = None
        if active_area:
            border_regions = self.analyze_border_regions(active_area)
        
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
            'detection_settings': {
                'sample_frames': int(self.sample_frames),
                'threshold': 10,  # Default threshold used
                'padding': 5      # Default padding used
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data
    
    def close(self):
        """Close video capture"""
        self.cap.release()


def detect_video_borders(video_path, output_dir=None, target_viz_time=150, search_window=120):
    """
    Main function to detect borders in a video file
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for results
        target_viz_time: Target time for visualization frame (seconds)
        search_window: Seconds to search around target time for good frame
    
    Returns:
        Dictionary with border detection results
    """
    video_path = Path(video_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = video_path.parent
        
    print(f"Processing: {video_path.name}")
    print("Detecting blanking borders and active picture area...")
    
    detector = VideoBorderDetector(video_path)
    
    # Detect borders
    active_area = detector.detect_blanking_borders(threshold=10)
    
    # Detect head switching artifacts (whether borders were found or not)
    head_switching_results = detector.detect_head_switching_artifacts(active_area)
    
    if active_area:
        print(f"\n✓ Active area detected: {active_area[2]}x{active_area[3]} at ({active_area[0]},{active_area[1]})")
        
        # Generate visualization
        viz_path = output_dir / f"{video_path.stem}_border_detection.jpg"
        detector.generate_border_visualization(viz_path, active_area, head_switching_results, target_viz_time, search_window)
        print(f"✓ Visualization saved: {viz_path}")
        
        # Save border data
        data_path = output_dir / f"{video_path.stem}_border_data.json"
        results = detector.save_border_data(data_path, active_area, head_switching_results)
        print(f"✓ Border data saved: {data_path}")
        
    else:
        print("⚠️ No clear borders detected")
        
        # Still generate visualization to show head switching artifacts
        viz_path = output_dir / f"{video_path.stem}_border_detection.jpg"
        detector.generate_border_visualization(viz_path, None, head_switching_results, target_viz_time, search_window)
        print(f"✓ Visualization saved: {viz_path}")
        
        results = detector.save_border_data(
            output_dir / f"{video_path.stem}_border_data.json",
            active_area, 
            head_switching_results
        )
        print(f"✓ Border data saved: {output_dir / f'{video_path.stem}_border_data.json'}")
    
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
    
    print(f"Using target time: {viz_time}s, search window: {search_window}s")
    results = detect_video_borders(video_file, target_viz_time=viz_time, search_window=search_window)
    
    if results['active_area']:
        x, y, w, h = results['active_area']
        print(f"\nBorder Detection Summary:")
        print(f"Active area: {w}x{h} at position ({x},{y})")
        print(f"Borders: L={x}px, R={results['video_properties']['width']-x-w}px, T={y}px, B={results['video_properties']['height']-y-h}px")
    else:
        print("\nNo borders detected - video appears to be full frame active content")
    
    # Head switching artifact summary
    if results.get('head_switching_artifacts'):
        hs_results = results['head_switching_artifacts']
        print(f"\nHead Switching Artifact Analysis:")
        print(f"Severity: {hs_results['severity']}")
        if hs_results['severity'] != 'none':
            print(f"Affected frames: {hs_results['frames_with_artifacts']}/{hs_results['frames_analyzed']} ({hs_results['artifact_percentage']:.1f}%)")
        else:
            print(f"✓ No significant head switching artifacts detected")