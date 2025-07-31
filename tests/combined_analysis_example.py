#!/usr/bin/env python3
"""
Combined Border Detection and FFprobe Analysis Example

This script shows how to use both border_detector.py and ffprobe_analyzer.py
together for comprehensive video analysis.
"""

from pathlib import Path
import sys

# Import the two modules
from border_detector import detect_video_borders
from ffmpeg_signalstats_analyzer import analyze_video_signalstats


def analyze_video_comprehensive(video_path, output_dir=None, start_time=120, duration=60, viz_time=150):
    """
    Perform comprehensive video analysis using both border detection and signalstats
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for all results
        start_time: Start analysis at this time in seconds
        duration: Duration of analysis in seconds
        viz_time: Target time for visualization frame
    """
    video_path = Path(video_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = video_path.parent
    
    print("="*80)
    print("COMPREHENSIVE VIDEO ANALYSIS")
    print("="*80)
    print(f"Video: {video_path.name}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Step 1: Border Detection
    print("STEP 1: BORDER DETECTION")
    print("-" * 40)
    border_results = detect_video_borders(video_path, output_dir, target_viz_time=viz_time)
    
    # Get the path to the border data file
    border_data_path = output_dir / f"{video_path.stem}_border_data.json"
    
    print()
    
    # Step 2: FFprobe Signalstats Analysis
    print("STEP 2: FFPROBE SIGNALSTATS ANALYSIS")
    print("-" * 40)
    signalstats_results = analyze_video_signalstats(
        video_path=video_path,
        border_data_path=border_data_path,
        output_dir=output_dir,
        start_time=start_time,
        duration=duration
    )
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Summary
    if border_results.get('active_area'):
        x, y, w, h = border_results['active_area']
        total_area = border_results['video_properties']['width'] * border_results['video_properties']['height']
        active_area_size = w * h
        active_percentage = (active_area_size / total_area) * 100
        
        print(f"Border Detection Summary:")
        print(f"  Active area: {w}x{h} ({active_percentage:.1f}% of frame)")
        print(f"  Borders detected: L={x}px, R={border_results['video_properties']['width']-x-w}px, T={y}px, B={border_results['video_properties']['height']-y-h}px")
    else:
        print("Border Detection: No borders detected")
    
    print(f"\nSignalstats Analysis: {signalstats_results.get('diagnosis', 'Analysis completed')}")
    
    print(f"\nOutput files generated:")
    print(f"  - {video_path.stem}_border_detection.jpg (visualization)")
    print(f"  - {video_path.stem}_border_data.json (border data)")
    print(f"  - {video_path.stem}_signalstats_analysis.json (complete analysis)")
    
    return {
        'border_results': border_results,
        'signalstats_results': signalstats_results
    }


def analyze_with_existing_borders(video_path, border_data_path, output_dir=None, 
                                 start_time=120, duration=60):
    """
    Analyze video using existing border data (skip border detection step)
    
    Args:
        video_path: Path to video file
        border_data_path: Path to existing border data JSON file
        output_dir: Output directory for results
        start_time: Start analysis at this time in seconds
        duration: Duration of analysis in seconds
    """
    print("="*80)
    print("SIGNALSTATS ANALYSIS WITH EXISTING BORDER DATA")
    print("="*80)
    
    return analyze_video_signalstats(
        video_path=video_path,
        border_data_path=border_data_path,
        output_dir=output_dir,
        start_time=start_time,
        duration=duration
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python combined_analysis_example.py <video_file> [viz_time]                    # Full analysis")
        print("  python combined_analysis_example.py <video_file> <border_data.json>           # Use existing borders")
        print()
        print("Examples:")
        print("  python combined_analysis_example.py video.mkv")
        print("  python combined_analysis_example.py video.mkv 180  # Use frame at 3 minutes for viz")
        print("  python combined_analysis_example.py video.mkv video_border_data.json")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    # Check if second argument is a JSON file (existing borders) or a time
    if len(sys.argv) > 2 and sys.argv[2].endswith('.json'):
        # Use existing border data
        border_data_file = sys.argv[2]
        results = analyze_with_existing_borders(video_file, border_data_file)
    else:
        # Full analysis - check if viz_time provided
        viz_time = 150  # default
        if len(sys.argv) > 2:
            try:
                viz_time = int(sys.argv[2])
            except ValueError:
                print(f"Warning: '{sys.argv[2]}' is not a valid time, using default 150s")
        
        results = analyze_video_comprehensive(video_file, viz_time=viz_time)
    
    print(f"\nAnalysis complete! Check the output directory for results.")