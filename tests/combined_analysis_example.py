#!/usr/bin/env python3
"""
Combined Border Detection, FFprobe Analysis, and Active Area BRNG Analysis

Updated to work with the revised BRNG analyzer that uses differential detection
and enhanced pattern recognition.
"""

from pathlib import Path
import sys
import json
import numpy as np

# Import the analysis modules
from border_detector import detect_video_borders
from ffmpeg_signalstats_analyzer import analyze_video_signalstats
from active_area_brng_analyzer import analyze_active_area_brng


def analyze_video_comprehensive(video_path, output_dir=None, start_time=120, duration=60, 
                               viz_time=150, brng_duration=300):
    """
    Perform comprehensive video analysis using border detection, signalstats, and active area BRNG
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for all results
        start_time: Start analysis at this time in seconds (for signalstats)
        duration: Duration of analysis in seconds (for signalstats)
        viz_time: Target time for visualization frame (for border detection)
        brng_duration: Duration to analyze for BRNG violations (seconds)
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
    
    # Step 3: Active Area BRNG Analysis (Updated for new analyzer)
    print("STEP 3: ACTIVE AREA BRNG ANALYSIS (DIFFERENTIAL DETECTION)")
    print("-" * 40)
    
    brng_results = None
    if border_results.get('active_area'):
        print("Analyzing BRNG violations in active picture area using differential detection...")
        brng_results = analyze_active_area_brng(
            video_path=video_path,
            border_data_path=border_data_path,
            output_dir=output_dir,
            duration_limit=brng_duration  # No longer needs sample_rate parameter
        )
    else:
        print("No border detection available - analyzing full frame for BRNG...")
        brng_results = analyze_active_area_brng(
            video_path=video_path,
            border_data_path=None,
            output_dir=output_dir,
            duration_limit=brng_duration
        )

    print()
    print("="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    
    # Combined Summary
    print("\n📊 COMBINED ANALYSIS RESULTS:")
    print("-" * 40)
    
    # Border Detection Summary
    if border_results.get('active_area'):
        x, y, w, h = border_results['active_area']
        total_area = border_results['video_properties']['width'] * border_results['video_properties']['height']
        active_area_size = w * h
        active_percentage = (active_area_size / total_area) * 100
        
        print(f"\n1. Border Detection:")
        print(f"   Active area: {w}x{h} ({active_percentage:.1f}% of frame)")
        print(f"   Borders: L={x}px, R={border_results['video_properties']['width']-x-w}px, T={y}px, B={border_results['video_properties']['height']-y-h}px")
        
        if border_results.get('head_switching_artifacts', {}).get('severity') != 'none':
            print(f"   ⚠️ Head switching artifacts: {border_results['head_switching_artifacts']['severity']}")
    else:
        print(f"\n1. Border Detection: No borders detected (full frame active)")
    
    # FFprobe Signalstats Summary (sample period)
    print(f"\n2. FFprobe Signalstats (sample: {start_time}s-{start_time+duration}s):")
    print(f"   {signalstats_results.get('diagnosis', 'Analysis completed')}")
    
    # Prefer active_area results when available, fall back to full_frame
    if signalstats_results.get('results', {}).get('active_area'):
        active_area = signalstats_results['results']['active_area']
        print(f"   Active area: {active_area['violation_percentage']:.1f}% frames, max {active_area['max_brng']:.4f}% pixels")
        
        # Also show border region summary if available
        if signalstats_results.get('results', {}).get('border_regions'):
            border_regions = signalstats_results['results']['border_regions']
            if border_regions:
                border_violations = [data['violation_percentage'] for data in border_regions.values() if data]
                if border_violations:
                    avg_border_violations = np.mean(border_violations)
                    print(f"   Border regions: {avg_border_violations:.1f}% frames average")
    elif signalstats_results.get('results', {}).get('full_frame'):
        full_frame = signalstats_results['results']['full_frame']
        print(f"   Full frame: {full_frame['violation_percentage']:.1f}% frames, max {full_frame['max_brng']:.4f}% pixels")
    
    # Active Area BRNG Analysis (Updated for new structure)
    print(f"\n3. Active Area BRNG Analysis (differential detection, up to {brng_duration}s):")
    if brng_results:
        # Get actionable report from new analyzer
        actionable_report = brng_results.get('actionable_report', {})
        analyzed_region = brng_results['video_info'].get('active_area') and 'active_area' or 'full_frame'
        
        print(f"   Method: Differential detection (eliminates false positives)")
        print(f"   Region analyzed: {analyzed_region}")
        print(f"   Assessment: {actionable_report.get('overall_assessment', 'Analysis complete')}")
        print(f"   Priority: {actionable_report.get('action_priority', 'none').upper()}")
        
        if brng_results['frames_with_violations'] > 0:
            print(f"   Frames with violations: {brng_results['frames_with_violations']}/{brng_results['total_frames_analyzed']}")
            print(f"   Maximum violation: {brng_results['max_violation_percentage']:.4f}% of pixels")
            
            # Show specific issues detected
            if actionable_report.get('recommendations'):
                issues = [rec['issue'] for rec in actionable_report['recommendations']]
                print(f"   Issues detected: {', '.join(issues[:3])}")
            
            # Show temporal pattern
            temporal = brng_results.get('temporal_analysis')
            if temporal:
                print(f"   Temporal pattern: {temporal.get('temporal_pattern', 'unknown')} - {temporal.get('interpretation', '')}")
        else:
            print(f"   ✔ No BRNG violations detected")
        
        # Mention thumbnails if saved
        if brng_results.get('saved_thumbnails'):
            print(f"   📸 Saved {len(brng_results['saved_thumbnails'])} diagnostic thumbnail(s)")
    else:
        print(f"   ⚠️ Analysis not available")
    
    # Cross-analysis insights
    print("\n" + "-" * 40)
    print("CROSS-ANALYSIS INSIGHTS:")
    
    # Compare FFprobe sample and detailed BRNG analysis
    ffprobe_data = None
    if signalstats_results.get('results', {}).get('active_area'):
        ffprobe_data = signalstats_results['results']['active_area']
    elif signalstats_results.get('results', {}).get('full_frame'):
        ffprobe_data = signalstats_results['results']['full_frame']
    
    if ffprobe_data and brng_results:
        ffprobe_max = ffprobe_data.get('max_brng', 0)
        detailed_max = brng_results.get('max_violation_percentage', 0)
        
        # Check if analyses agree
        if ffprobe_max < 0.01 and detailed_max < 0.01:
            print("✔ Both analyses confirm minimal BRNG violations")
            print("   → Video appears broadcast-safe")
        elif ffprobe_max > 0.1 or detailed_max > 0.1:
            print("⚠️ Both analyses detect notable BRNG violations")
            
            # Show specific recommendations from new analyzer
            if brng_results.get('actionable_report', {}).get('recommendations'):
                print("   Specific issues:")
                for rec in brng_results['actionable_report']['recommendations'][:2]:
                    severity = rec.get('severity', 'unknown')
                    issue = rec.get('issue', 'Unknown')
                    print(f"   • {issue} ({severity} severity)")
            
            print("   → Review recommended for broadcast compliance")
        else:
            print("ℹ️ Minor BRNG violations detected")
            print("   → Likely acceptable for broadcast")
    
    # Enhanced insights based on new analyzer's pattern detection
    if brng_results and brng_results.get('aggregate_patterns'):
        patterns = brng_results['aggregate_patterns']
        
        if patterns.get('primary_violation_zone'):
            zone = patterns['primary_violation_zone']
            if zone == 'highlights':
                print("\n⚠️ Pattern Analysis: Violations primarily in highlight areas")
                print("   → Consider reducing video levels or applying highlight compression")
            elif zone == 'shadows':
                print("\n⚠️ Pattern Analysis: Violations primarily in shadow areas")
                print("   → Consider lifting blacks or adjusting shadow detail")
        
        if patterns.get('horizontal_banding_frames', 0) > 0:
            print("\n⚠️ Pattern Analysis: Horizontal banding detected")
            print("   → Check deinterlacing settings or field order")
        
        if patterns.get('avg_edge_violation_ratio', 0) > 0.5:
            print("\n⚠️ Pattern Analysis: Edge enhancement artifacts detected")
            print("   → Consider reducing sharpening in processing")
    
    # Active area vs borders insight
    if (border_results.get('active_area') and 
        signalstats_results.get('results', {}).get('border_regions')):
        
        print("\n✔ Analysis focused on active picture content")
        print("   → Border/blanking areas excluded from violation assessment")
        
        # If borders have high violations but active area is clean
        border_regions = signalstats_results['results']['border_regions']
        if border_regions:
            border_brngs = [data.get('avg_brng', 0) for data in border_regions.values() if data]
            if border_brngs and max(border_brngs) > 10:
                print("   Note: High BRNG values in borders (expected for blanking)")
    
    print("\n" + "-" * 40)
    print(f"\nOutput files generated:")
    print(f"  - {video_path.stem}_border_detection.jpg (border visualization)")
    print(f"  - {video_path.stem}_border_data.json (border detection data)")
    print(f"  - {video_path.stem}_signalstats_analysis.json (FFprobe analysis)")
    if brng_results:
        print(f"  - {video_path.stem}_active_brng_analysis.json (enhanced BRNG analysis)")
        if brng_results.get('saved_thumbnails'):
            print(f"  - brng_thumbnails/ ({len(brng_results['saved_thumbnails'])} diagnostic thumbnails)")
    
    # Save combined summary (updated structure)
    summary_path = output_dir / f"{video_path.stem}_combined_summary.json"
    
    # Get signalstats summary data
    signalstats_summary_data = {}
    if signalstats_results.get('results', {}).get('active_area'):
        signalstats_summary_data['active_area_violations'] = signalstats_results['results']['active_area'].get('violation_percentage')
        signalstats_summary_data['active_area_max_brng'] = signalstats_results['results']['active_area'].get('max_brng')
    if signalstats_results.get('results', {}).get('full_frame'):
        signalstats_summary_data['full_frame_violations'] = signalstats_results['results']['full_frame'].get('violation_percentage')
        signalstats_summary_data['full_frame_max_brng'] = signalstats_results['results']['full_frame'].get('max_brng')
    
    # Extract key info from new BRNG analyzer
    brng_summary_data = {}
    if brng_results:
        actionable = brng_results.get('actionable_report', {})
        brng_summary_data = {
            'analyzed': True,
            'method': 'differential_detection',
            'region': analyzed_region,
            'frames_analyzed': brng_results.get('total_frames_analyzed', 0),
            'frames_with_violations': brng_results.get('frames_with_violations', 0),
            'max_violation_percentage': brng_results.get('max_violation_percentage', 0),
            'assessment': actionable.get('overall_assessment', 'Unknown'),
            'priority': actionable.get('action_priority', 'none'),
            'issues_detected': [rec['issue'] for rec in actionable.get('recommendations', [])],
            'temporal_pattern': brng_results.get('temporal_analysis', {}).get('temporal_pattern', 'unknown') if brng_results.get('temporal_analysis') else None,
            'thumbnails_saved': len(brng_results.get('saved_thumbnails', []))
        }
    else:
        brng_summary_data = {'analyzed': False}
    
    combined_results = {
        'video_file': str(video_path),
        'analyses_performed': {
            'border_detection': True,
            'ffprobe_signalstats': True,
            'active_area_brng_differential': brng_results is not None
        },
        'border_results_summary': {
            'has_borders': border_results.get('active_area') is not None,
            'active_area': border_results.get('active_area'),
            'head_switching': border_results.get('head_switching_artifacts', {}).get('severity', 'none')
        },
        'signalstats_summary': {
            'diagnosis': signalstats_results.get('diagnosis'),
            **signalstats_summary_data
        },
        'active_brng_summary': brng_summary_data
    }
    
    with open(summary_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"  - {video_path.stem}_combined_summary.json (combined analysis summary)")
    
    return {
        'border_results': border_results,
        'signalstats_results': signalstats_results,
        'active_brng_results': brng_results
    }


def analyze_with_existing_borders(video_path, border_data_path, output_dir=None, 
                                 start_time=120, duration=60, brng_duration=300):
    """
    Analyze video using existing border data (skip border detection step)
    
    Args:
        video_path: Path to video file
        border_data_path: Path to existing border data JSON file
        output_dir: Output directory for results
        start_time: Start analysis at this time in seconds
        duration: Duration of analysis in seconds
        brng_duration: Duration for BRNG analysis
    """
    video_path = Path(video_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = video_path.parent
    
    print("="*80)
    print("ANALYSIS WITH EXISTING BORDER DATA")
    print("="*80)
    
    # Load existing border data
    with open(border_data_path, 'r') as f:
        border_results = json.load(f)
    
    print(f"✔ Loaded border data from: {border_data_path}")
    
    # Run signalstats analysis
    print("\nRunning FFprobe signalstats analysis...")
    signalstats_results = analyze_video_signalstats(
        video_path=video_path,
        border_data_path=border_data_path,
        output_dir=output_dir,
        start_time=start_time,
        duration=duration
    )
    
    # Run active area BRNG analysis (updated for new analyzer)
    print("\nRunning enhanced active area BRNG analysis with differential detection...")
    brng_results = analyze_active_area_brng(
        video_path=video_path,
        border_data_path=border_data_path,
        output_dir=output_dir,
        duration_limit=brng_duration  # No longer needs sample_rate
    )
    
    return {
        'border_results': border_results,
        'signalstats_results': signalstats_results,
        'active_brng_results': brng_results
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python combined_analysis.py <video_file> [options]")
        print()
        print("Options:")
        print("  [viz_time]              - Time in seconds for border detection frame (default: 150)")
        print("  [border_data.json]      - Use existing border data file")
        print()
        print("Examples:")
        print("  python combined_analysis.py video.mkv")
        print("  python combined_analysis.py video.mkv 180")
        print("  python combined_analysis.py video.mkv video_border_data.json")
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
    
    print(f"\n✔ Analysis complete! Check the output directory for results.")