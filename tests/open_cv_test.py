import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if hasattr(obj, 'dtype'):
            # Handle all numpy types
            if np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            elif np.issubdtype(obj.dtype, np.floating):
                return float(obj)
            else:
                return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class AnalogVideoBorderDetector:
    """
    Specialized border detector for analog video captures with blanking period artifacts.
    Focuses on detecting consistent left/right borders while being tolerant of top/bottom variations.
    """
    
    def __init__(self, video_path, sample_frames=30):
        self.video_path = video_path
        self.sample_frames = sample_frames
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def detect_blanking_borders(self, threshold=10, edge_sample_width=50):
        """
        Detect borders from analog blanking periods, focusing on left/right edges.
        
        Args:
            threshold: Luma threshold for black detection
            edge_sample_width: Width of edge region to sample for more robust detection
            
        Returns:
            tuple: (x, y, width, height) of active area
        """
        frame_indices = np.linspace(0, self.total_frames - 1, 
                                   min(self.sample_frames, self.total_frames), 
                                   dtype=int)
        
        left_borders = []
        right_borders = []
        top_borders = []
        bottom_borders = []
        
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            
            if not ret:
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Detect LEFT border - scan from left to right
            left = 0
            for x in range(min(edge_sample_width, w)):
                # Check a vertical strip for any pixels above threshold
                column_max = np.max(gray[:, x])
                if column_max > threshold:
                    left = x
                    break
                    
            # Detect RIGHT border - scan from right to left
            right = w
            for x in range(w-1, max(w-edge_sample_width, -1), -1):
                column_max = np.max(gray[:, x])
                if column_max > threshold:
                    right = x + 1
                    break
                    
            # For top/bottom, use center region to avoid edge artifacts
            center_start = w // 4
            center_end = 3 * w // 4
            
            # Detect TOP border using center region
            top = 0
            for y in range(min(10, h)):  # Only check first 10 lines
                if np.max(gray[y, center_start:center_end]) > threshold:
                    top = y
                    break
                    
            # Detect BOTTOM border using center region
            bottom = h
            for y in range(h-1, max(h-10, -1), -1):  # Only check last 10 lines
                if np.max(gray[y, center_start:center_end]) > threshold:
                    bottom = y + 1
                    break
                    
            left_borders.append(left)
            right_borders.append(right)
            top_borders.append(top)
            bottom_borders.append(bottom)
            
        if not left_borders:
            return None
            
        # Calculate stable borders using median and mode
        # For left/right borders, use median for stability
        median_left = int(np.median(left_borders))
        median_right = int(np.median(right_borders))
        
        # For top/bottom, use mode (most common value) to handle variations
        top_unique, top_counts = np.unique(top_borders, return_counts=True)
        mode_top = int(top_unique[np.argmax(top_counts)])
        
        bottom_unique, bottom_counts = np.unique(bottom_borders, return_counts=True)
        mode_bottom = int(bottom_unique[np.argmax(bottom_counts)])
        
        # Calculate active area
        active_width = median_right - median_left
        active_height = mode_bottom - mode_top
        
        # Validate results
        if active_width < 100 or active_height < 100:  # Sanity check
            print("Warning: Detected active area seems too small")
            return None
            
        result = (median_left, mode_top, active_width, active_height)
        
        # Print detection statistics
        print(f"\nBorder detection statistics:")
        print(f"  Left border: median={median_left}, std={np.std(left_borders):.1f}")
        print(f"  Right border: median={median_right}, std={np.std(right_borders):.1f}")
        print(f"  Top border: mode={mode_top}, variations={len(top_unique)}")
        print(f"  Bottom border: mode={mode_bottom}, variations={len(bottom_unique)}")
        print(f"  Active area: {active_width}x{active_height} at ({median_left},{mode_top})")
        
        return result
        
    def analyze_broadcast_range_detailed(self, active_area=None):
        """
        Enhanced broadcast range analysis with detailed statistics
        """
        results = {
            'total_frames': self.total_frames,
            'video_dimensions': f"{self.width}x{self.height}",
            'active_area': list(active_area) if active_area else None,
            'frames_analyzed': 0,
            'frames_with_violations': 0,
            'violation_pixels_per_frame': [],
            'violation_types': {
                'super_black': 0,  # Y < 16
                'super_white': 0,  # Y > 235
                'illegal_chroma': 0  # U/V outside 16-240
            }
        }
        
        # Sample every Nth frame for efficiency
        sample_interval = max(1, self.total_frames // 1000)  # Analyze ~1000 frames max
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for frame_idx in range(0, self.total_frames, sample_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            results['frames_analyzed'] += 1
            
            # Extract active area if specified
            if active_area:
                x, y, w, h = active_area
                roi = frame[y:y+h, x:x+w]
            else:
                roi = frame
                
            # Convert to YUV
            yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            u_channel = yuv[:, :, 1]
            v_channel = yuv[:, :, 2]
            
            # Count violations
            super_black = int(np.sum(y_channel < 16))
            super_white = int(np.sum(y_channel > 235))
            illegal_u = int(np.sum((u_channel < 16) | (u_channel > 240)))
            illegal_v = int(np.sum((v_channel < 16) | (v_channel > 240)))
            
            total_violations = super_black + super_white + illegal_u + illegal_v
            
            if total_violations > 0:
                results['frames_with_violations'] += 1
                results['violation_pixels_per_frame'].append(total_violations)
                
                if super_black > 0:
                    results['violation_types']['super_black'] += 1
                if super_white > 0:
                    results['violation_types']['super_white'] += 1
                if illegal_u > 0 or illegal_v > 0:
                    results['violation_types']['illegal_chroma'] += 1
                    
        # Calculate statistics
        if results['frames_analyzed'] > 0:
            results['violation_percentage'] = float(results['frames_with_violations']) / float(results['frames_analyzed']) * 100.0
            
            if results['violation_pixels_per_frame']:
                results['avg_violation_pixels'] = float(np.mean(results['violation_pixels_per_frame']))
                results['max_violation_pixels'] = max(results['violation_pixels_per_frame'])
            else:
                results['avg_violation_pixels'] = 0
                results['max_violation_pixels'] = 0
                
        return results
        
    def generate_comparison_report(self, output_path, active_area=None):
        """
        Generate visual comparison showing full frame vs active area
        """
        # Get a representative frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames // 2)
        ret, frame = self.cap.read()
        
        if not ret:
            return False
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Full frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax1.imshow(frame_rgb)
        ax1.set_title('Full Frame (with blanking artifacts)')
        ax1.axis('off')
        
        # Add rectangle showing active area
        if active_area:
            x, y, w, h = active_area
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='lime', facecolor='none')
            ax1.add_patch(rect)
            
            # Active area only
            active_frame = frame_rgb[y:y+h, x:x+w]
            ax2.imshow(active_frame)
            ax2.set_title('Active Picture Area Only')
            ax2.axis('off')
            
            # Add text annotations
            fig.text(0.5, 0.02, 
                    f'Detected borders: Left={x}px, Right={self.width-x-w}px, Top={y}px, Bottom={self.height-y-h}px',
                    ha='center', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No borders detected', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    def close(self):
        self.cap.release()


def process_analog_video(video_path, output_dir=None):
    """
    Main processing function for analog video with blanking artifacts
    """
    video_path = Path(video_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = video_path.parent
        
    print(f"Processing analog video: {video_path.name}")
    print(f"Expected: 4:3 content with blanking period artifacts on sides")
    
    detector = AnalogVideoBorderDetector(video_path)
    
    # Step 1: Detect borders with parameters tuned for analog video
    print("\nDetecting active picture area (excluding blanking)...")
    active_area = detector.detect_blanking_borders(threshold=10)
    
    # Step 2: Analyze full frame
    print("\nAnalyzing full frame (including blanking areas)...")
    full_results = detector.analyze_broadcast_range_detailed(active_area=None)
    
    # Step 3: Analyze active area only
    active_results = None
    if active_area:
        print("\nAnalyzing active picture area only...")
        active_results = detector.analyze_broadcast_range_detailed(active_area=active_area)
    
    # Step 4: Generate visual comparison
    comparison_path = output_dir / f"{video_path.stem}_active_area_comparison.jpg"
    detector.generate_comparison_report(comparison_path, active_area)
    print(f"\nVisual comparison saved: {comparison_path}")
    
    # Step 5: Create comprehensive report
    report = {
        'video_file': str(video_path),
        'video_type': 'Analog capture with blanking artifacts',
        'active_area': list(active_area) if active_area else None,
        'active_area_detection': {
            'detected': active_area is not None,
            'coordinates': {
                'x': active_area[0] if active_area else None,
                'y': active_area[1] if active_area else None,
                'width': active_area[2] if active_area else None,
                'height': active_area[3] if active_area else None
            } if active_area else None,
            'blanking_widths': {
                'left': active_area[0] if active_area else 0,
                'right': detector.width - active_area[0] - active_area[2] if active_area else 0,
                'top': active_area[1] if active_area else 0,
                'bottom': detector.height - active_area[1] - active_area[3] if active_area else 0
            } if active_area else None
        },
        'broadcast_range_analysis': {
            'full_frame': full_results,
            'active_area_only': active_results if active_results else None
        },
        'diagnosis': ''
    }
    
    # Determine diagnosis
    if active_area and active_results:
        full_violations = full_results['violation_percentage']
        active_violations = active_results['violation_percentage']
        
        if full_violations > 10 and active_violations < 2:
            report['diagnosis'] = "✓ Violations are in blanking areas only (expected for analog captures)"
            report['action_required'] = False
        elif active_violations > 5:
            report['diagnosis'] = "⚠️ Broadcast violations found in active picture content"
            report['action_required'] = True
            
            # Add details about violation types
            dominant_violation = max(active_results['violation_types'].items(), 
                                   key=lambda x: x[1])[0]
            report['dominant_violation_type'] = dominant_violation
        else:
            report['diagnosis'] = "✓ Minimal violations - file appears to be within broadcast range"
            report['action_required'] = False
    else:
        report['diagnosis'] = "Unable to detect consistent borders"
        report['action_required'] = True
    
    # Save JSON report
    json_path = output_dir / f"{video_path.stem}_analog_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"Analysis report saved: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Video: {detector.width}x{detector.height}")
    
    if active_area:
        x, y, w, h = active_area
        print(f"Active picture area: {w}x{h}")
        print(f"Blanking on sides: {x}px left, {detector.width-x-w}px right")
        print(f"Blanking top/bottom: {y}px top, {detector.height-y-h}px bottom")
    
    print(f"\n{report['diagnosis']}")
    
    if active_results and full_results:
        reduction = full_results['violation_percentage'] - active_results['violation_percentage']
        if reduction > 5:
            print(f"Excluding blanking areas reduced violations by {reduction:.1f}%")
    
    detector.close()
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = "JPC_AV_00011.mkv"
        
    results = process_analog_video(video_file)