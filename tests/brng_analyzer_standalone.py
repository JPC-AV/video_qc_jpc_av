#!/usr/bin/env python3
"""
Standalone BRNG Violation Analyzer for QCTools Reports

This script analyzes QCTools XML reports to identify and summarize
broadcast range (BRNG) violations in video files.

Usage:
    python brng_analyzer.py <video_file.mkv>
"""

import gzip
import os
import sys
import csv
import io
import collections
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


@dataclass
class BrngSegment:
    """Data class for BRNG violation segments"""
    start_time: float
    end_time: float
    start_timestamp: str
    end_timestamp: str
    violation_percentage: float
    violation_count: int
    total_frames: int
    max_brng: float
    avg_brng: float


def load_etree():
    """Helper function to load lxml.etree with error handling"""
    try:
        from lxml import etree
        return etree
    except ImportError as e:
        print(f"Error: lxml library required. Install with: pip install lxml")
        print(f"Error details: {e}")
        return None


def find_qctools_report(video_path: str) -> Optional[str]:
    """
    Search for qctools report files related to the video.
    
    Searches in:
    1. Same directory as video
    2. {video_id}_qc_metadata folder
    3. {video_id}_vrecord_metadata folder
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Path to qctools report if found, None otherwise
    """
    video_path = Path(video_path)
    video_dir = video_path.parent
    video_id = video_path.stem
    
    # Remove common extensions from video_id if present
    for ext in ['.mkv', '.mov', '.mp4', '.avi', '.mxf']:
        if video_id.endswith(ext):
            video_id = video_id[:-len(ext)]
    
    # Define search locations
    search_locations = [
        # Same directory as video
        video_dir,
        # Metadata folders
        video_dir / f"{video_id}_qc_metadata",
        video_dir / f"{video_id}_vrecord_metadata"
    ]
    
    # Search patterns for qctools files
    patterns = [
        f"{video_id}.qctools.xml.gz",
        f"{video_id}.qctools.mkv",
        f"{video_id}*.qctools.xml.gz",
        f"{video_id}*.qctools.mkv",
        "*.qctools.xml.gz",
        "*.qctools.mkv"
    ]
    
    # Search each location
    for location in search_locations:
        if location.exists() and location.is_dir():
            for pattern in patterns:
                matches = list(location.glob(pattern))
                if matches:
                    # Prefer exact matches
                    for match in matches:
                        if video_id in match.stem:
                            print(f"✓ Found QCTools report: {match}")
                            return str(match)
                    # Return first match if no exact match
                    print(f"✓ Found QCTools report: {matches[0]}")
                    return str(matches[0])
    
    return None


def safe_gzip_open_with_encoding_fallback(file_path: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Opens a gzipped file with encoding fallback handling.
    
    Returns:
        tuple: (raw_bytes, encoding_used) or (None, None) if failed
    """
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    try:
        with gzip.open(file_path, 'rb') as gz_file:
            raw_content = gz_file.read()
    except Exception as e:
        print(f"Error reading gzipped file {file_path}: {e}")
        return None, None
    
    # Try to decode with different encodings
    for encoding in encodings_to_try:
        try:
            raw_content.decode(encoding)
            return raw_content, encoding
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try with error handling
    try:
        raw_content.decode('utf-8', errors='replace')
        return raw_content, 'utf-8-replace'
    except Exception as e:
        print(f"Failed to decode {file_path}: {e}")
        return None, None


def safe_gzip_iterparse(file_path: str, etree_module):
    """
    Safely parse gzipped XML with encoding fallback.
    
    Returns:
        iterator: XML parser iterator or None if failed
    """
    raw_content, encoding_used = safe_gzip_open_with_encoding_fallback(file_path)
    
    if raw_content is None:
        return None
    
    try:
        bytes_io = io.BytesIO(raw_content)
        parser_iter = etree_module.iterparse(bytes_io, events=('end',), tag='frame')
        return parser_iter
    except Exception as e:
        print(f"Error creating XML parser: {e}")
        return None


def extract_report_from_mkv(mkv_path: str) -> Optional[str]:
    """
    Extract qctools.xml.gz report from QCTools MKV file.
    
    Args:
        mkv_path: Path to .qctools.mkv file
        
    Returns:
        Path to extracted .qctools.xml.gz file or None if failed
    """
    import subprocess
    
    output_path = mkv_path.replace(".qctools.mkv", ".qctools.xml.gz")
    
    # Remove existing report if present
    if os.path.isfile(output_path):
        os.remove(output_path)
    
    # Extract XML report from MKV
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'panic',
        '-dump_attachment:t:0', output_path,
        '-i', mkv_path
    ]
    
    print(f"Extracting report from MKV file...")
    try:
        subprocess.run(cmd, check=True)
        if os.path.isfile(output_path):
            print(f"✓ Extracted report to: {output_path}")
            return output_path
    except subprocess.CalledProcessError:
        print(f"✗ Failed to extract report from MKV")
    except FileNotFoundError:
        print(f"✗ ffmpeg not found. Please install ffmpeg.")
    
    return None


def dts2ts(frame_pkt_dts_time: str) -> str:
    """
    Converts time in seconds to HH:MM:SS.ssss format.
    
    Args:
        frame_pkt_dts_time: Time in seconds as string
        
    Returns:
        Formatted time string
    """
    seconds = float(frame_pkt_dts_time)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    
    hours = f"{int(hours):02d}"
    minutes = f"{int(minutes):02d}"
    seconds_str = f"{seconds:06.3f}"
    
    return f"{hours}:{minutes}:{seconds_str}"


def analyze_brng_violations(report_path: str, window_size: int = 30, 
                           brng_threshold: float = 0.01) -> List[BrngSegment]:
    """
    Analyze BRNG violations in QCTools report.
    
    Args:
        report_path: Path to qctools.xml.gz or qctools.mkv file
        window_size: Size of analysis window in seconds
        brng_threshold: BRNG value above which is considered a violation
        
    Returns:
        List of BrngSegment objects with violation data
    """
    etree = load_etree()
    if etree is None:
        return []
    
    # Handle MKV files
    if report_path.endswith('.qctools.mkv'):
        extracted_path = extract_report_from_mkv(report_path)
        if extracted_path:
            report_path = extracted_path
        else:
            return []
    
    # Determine packet timestamp field
    pkt = None
    with gzip.open(report_path) as xml:
        for event, elem in etree.iterparse(xml, events=('end',), tag='frame'):
            if elem.attrib.get('media_type') == "video":
                import re
                match = re.search(r"pkt_.ts_time", etree.tostring(elem).decode('utf-8'))
                if match:
                    pkt = match.group()
                    break
                elem.clear()
    
    if not pkt:
        print("✗ Could not determine timestamp field in QCTools report")
        return []
    
    # Parse report for BRNG violations
    parser_iter = safe_gzip_iterparse(report_path, etree)
    if parser_iter is None:
        return []
    
    # Collect frame data
    frames_data = []
    total_frames = 0
    
    print("\nAnalyzing BRNG violations...")
    try:
        for event, elem in parser_iter:
            if elem.attrib.get('media_type') == "video":
                total_frames += 1
                frame_time = float(elem.attrib[pkt])
                
                # Extract BRNG value
                brng_value = None
                for tag in list(elem):
                    if tag.attrib.get('key', '').endswith('BRNG'):
                        brng_value = float(tag.attrib['value'])
                        break
                
                if brng_value is not None:
                    frames_data.append({
                        'time': frame_time,
                        'timestamp': dts2ts(elem.attrib[pkt]),
                        'brng': brng_value,
                        'has_violation': brng_value > brng_threshold
                    })
                
                elem.clear()
                
                # Progress indicator
                if total_frames % 1000 == 0:
                    print(f"  Processed {total_frames} frames...", end='\r')
        
        print(f"  Processed {total_frames} frames total    ")
        
    except Exception as e:
        print(f"Error parsing report: {e}")
        return []
    
    if not frames_data:
        print("✗ No frame data found")
        return []
    
    # Group frames into windows
    windows = {}
    for frame in frames_data:
        window_start = int(frame['time'] // window_size) * window_size
        
        if window_start not in windows:
            windows[window_start] = {
                'frames': [],
                'violations': 0,
                'max_brng': 0,
                'brng_values': []
            }
        
        windows[window_start]['frames'].append(frame)
        if frame['has_violation']:
            windows[window_start]['violations'] += 1
            windows[window_start]['max_brng'] = max(
                windows[window_start]['max_brng'],
                frame['brng']
            )
        windows[window_start]['brng_values'].append(frame['brng'])
    
    # Create segment objects
    segments = []
    for start_time, data in windows.items():
        if data['violations'] > 0:  # Only include windows with violations
            end_time = start_time + window_size
            
            # Get actual time range from frames
            if data['frames']:
                actual_start = min(f['time'] for f in data['frames'])
                actual_end = max(f['time'] for f in data['frames'])
                start_timestamp = min(f['timestamp'] for f in data['frames'])
                end_timestamp = max(f['timestamp'] for f in data['frames'])
            else:
                actual_start = start_time
                actual_end = end_time
                start_timestamp = dts2ts(str(start_time))
                end_timestamp = dts2ts(str(end_time))
            
            segment = BrngSegment(
                start_time=actual_start,
                end_time=actual_end,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                violation_percentage=(data['violations'] / len(data['frames'])) * 100,
                violation_count=data['violations'],
                total_frames=len(data['frames']),
                max_brng=data['max_brng'],
                avg_brng=sum(data['brng_values']) / len(data['brng_values']) if data['brng_values'] else 0
            )
            segments.append(segment)
    
    # Sort by violation percentage (highest first)
    segments.sort(key=lambda x: x.violation_percentage, reverse=True)
    
    return segments


def save_results(segments: List[BrngSegment], output_path: str, video_path: str):
    """
    Save BRNG analysis results to CSV file.
    
    Args:
        segments: List of BrngSegment objects
        output_path: Path for output CSV
        video_path: Original video path for reference
    """
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(["BRNG Violation Analysis Report"])
        writer.writerow([f"Video: {os.path.basename(video_path)}"])
        writer.writerow([f"Report generated: {output_path}"])
        writer.writerow([])
        
        if not segments:
            writer.writerow(["No BRNG violations detected"])
            return
        
        # Summary statistics
        total_violations = sum(s.violation_count for s in segments)
        total_frames_checked = sum(s.total_frames for s in segments)
        max_brng_overall = max(s.max_brng for s in segments)
        
        writer.writerow(["Summary Statistics"])
        writer.writerow([f"Total segments with violations: {len(segments)}"])
        writer.writerow([f"Total frames with violations: {total_violations}"])
        writer.writerow([f"Maximum BRNG value: {max_brng_overall:.6f}"])
        writer.writerow([])
        
        # Detailed segments
        writer.writerow(["Top Violation Segments (sorted by violation percentage)"])
        writer.writerow(["Start Time", "End Time", "Duration (s)", "Violation %", 
                        "Violations/Total", "Max BRNG", "Avg BRNG"])
        
        for segment in segments[:20]:  # Top 20 segments
            duration = segment.end_time - segment.start_time
            writer.writerow([
                segment.start_timestamp,
                segment.end_timestamp,
                f"{duration:.1f}",
                f"{segment.violation_percentage:.1f}%",
                f"{segment.violation_count}/{segment.total_frames}",
                f"{segment.max_brng:.6f}",
                f"{segment.avg_brng:.6f}"
            ])


def print_summary(segments: List[BrngSegment], top_n: int = 10):
    """
    Print summary of BRNG violations to console.
    
    Args:
        segments: List of BrngSegment objects
        top_n: Number of top segments to display
    """
    print("\n" + "="*80)
    print("BRNG VIOLATION ANALYSIS SUMMARY")
    print("="*80)
    
    if not segments:
        print("✓ No BRNG violations detected")
        return
    
    # Overall statistics
    total_violations = sum(s.violation_count for s in segments)
    total_frames = sum(s.total_frames for s in segments)
    max_brng = max(s.max_brng for s in segments)
    
    print(f"\nOverall Statistics:")
    print(f"  Segments with violations: {len(segments)}")
    print(f"  Total frames analyzed: {total_frames}")
    print(f"  Frames with violations: {total_violations} ({(total_violations/total_frames)*100:.1f}%)")
    print(f"  Maximum BRNG value: {max_brng:.6f}")
    
    # Top violation segments
    print(f"\nTop {min(top_n, len(segments))} Segments with Highest Violation Rates:")
    print("-" * 80)
    
    for i, segment in enumerate(segments[:top_n], 1):
        duration = segment.end_time - segment.start_time
        print(f"\n{i}. Time: {segment.start_timestamp} - {segment.end_timestamp} ({duration:.1f}s)")
        print(f"   Violation rate: {segment.violation_percentage:.1f}%")
        print(f"   Frames: {segment.violation_count}/{segment.total_frames} violations")
        print(f"   BRNG values: max={segment.max_brng:.6f}, avg={segment.avg_brng:.6f}")
    
    # Identify patterns
    print("\n" + "-" * 80)
    print("Pattern Analysis:")
    
    # Check if violations are concentrated at start/end
    if segments:
        video_duration = max(s.end_time for s in segments)
        start_violations = [s for s in segments if s.start_time < 60]
        end_violations = [s for s in segments if s.start_time > video_duration - 60]
        
        if len(start_violations) > len(segments) * 0.3:
            print("  ⚠ High concentration of violations at video start")
        if len(end_violations) > len(segments) * 0.3:
            print("  ⚠ High concentration of violations at video end")
        
        # Check severity
        severe_segments = [s for s in segments if s.violation_percentage > 50]
        moderate_segments = [s for s in segments if 10 < s.violation_percentage <= 50]
        minor_segments = [s for s in segments if s.violation_percentage <= 10]
        
        if severe_segments:
            print(f"  ⚠ {len(severe_segments)} segments with severe violations (>50%)")
        if moderate_segments:
            print(f"  ⚠ {len(moderate_segments)} segments with moderate violations (10-50%)")
        if minor_segments:
            print(f"  ℹ {len(minor_segments)} segments with minor violations (<10%)")


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python brng_analyzer.py <video_file>")
        print("\nThis script analyzes BRNG violations in QCTools reports.")
        print("It will search for the QCTools report in:")
        print("  - Same directory as the video")
        print("  - {video_id}_qc_metadata folder")
        print("  - {video_id}_vrecord_metadata folder")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"Processing: {os.path.basename(video_path)}")
    
    # Find QCTools report
    report_path = find_qctools_report(video_path)
    
    if not report_path:
        print("\n✗ No QCTools report found!")
        print("  Searched for .qctools.xml.gz and .qctools.mkv files")
        print("  Please ensure QCTools has been run on this video")
        sys.exit(1)
    
    # Analyze BRNG violations
    segments = analyze_brng_violations(report_path)
    
    # Print summary
    print_summary(segments)
    
    # Save detailed results
    video_path_obj = Path(video_path)
    output_dir = video_path_obj.parent
    output_file = output_dir / f"{video_path_obj.stem}_brng_analysis.csv"
    
    save_results(segments, str(output_file), video_path)
    
    if segments:
        print(f"\n✓ Detailed results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()