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
                               viz_time=150, brng_duration=300,
                               # Border detection parameters
                               border_threshold=10, border_edge_sample_width=100, 
                               border_sample_frames=30, border_padding=5,
                               # Auto-retry settings
                               auto_retry_borders=True, boundary_artifact_threshold=10):  # Lowered from 20
    """
    Perform comprehensive video analysis with intelligent border re-detection
    
    Workflow:
    1. Detect borders
    2. Run BRNG analysis
    3. If BRNG finds boundary artifacts, re-run border detection with adjusted parameters
    4. Re-run BRNG analysis if borders were adjusted
    5. Run signalstats with final borders
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for all results
        start_time: Start analysis at this time in seconds (for signalstats)
        duration: Duration of analysis in seconds (for signalstats)
        viz_time: Target time for visualization frame (for border detection)
        brng_duration: Duration to analyze for BRNG violations (seconds)
        border_threshold: Brightness threshold for border detection (default: 10)
        border_edge_sample_width: How far from edges to scan (default: 100)
        border_sample_frames: Number of frames to sample (default: 30)
        border_padding: Extra pixels to add for tighter active area (default: 5)
        auto_retry_borders: Automatically retry border detection if artifacts found (default: True)
        boundary_artifact_threshold: Percentage of frames with artifacts to trigger retry (default: 10%)
    """
    video_path = Path(video_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = video_path.parent
    
    print("="*80)
    print("COMPREHENSIVE VIDEO ANALYSIS WITH INTELLIGENT BORDER DETECTION")
    print("="*80)
    print(f"Video: {video_path.name}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Step 1: Initial Border Detection
    print("STEP 1: INITIAL BORDER DETECTION")
    print("-" * 40)
    
    border_results = detect_video_borders(
        video_path, 
        output_dir, 
        target_viz_time=viz_time,
        search_window=120,
        threshold=border_threshold,
        edge_sample_width=border_edge_sample_width,
        sample_frames=border_sample_frames,
        padding=border_padding
    )
    
    # Get the path to the border data file
    border_data_path = output_dir / f"{video_path.stem}_border_data.json"
    
    print()
    
    # Step 2: Initial BRNG Analysis
    print("STEP 2: INITIAL ACTIVE AREA BRNG ANALYSIS")
    print("-" * 40)
    
    brng_results = None
    if border_results.get('active_area'):
        print("Analyzing BRNG violations in detected active picture area...")
        brng_results = analyze_active_area_brng(
            video_path=video_path,
            border_data_path=border_data_path,
            output_dir=output_dir,
            duration_limit=brng_duration
        )
    else:
        print("No borders detected - analyzing full frame for BRNG...")
        brng_results = analyze_active_area_brng(
            video_path=video_path,
            border_data_path=None,
            output_dir=output_dir,
            duration_limit=brng_duration
        )
    
    print()
    
    # Step 3: Check if BRNG found boundary artifacts and retry if needed
    border_retry_performed = False
    if auto_retry_borders and brng_results:
        aggregate = brng_results.get('aggregate_patterns', {})
        
        # Check for the clear signal from improved analyzer
        requires_adjustment = aggregate.get('requires_border_adjustment', False)
        
        # Also check percentages as fallback
        boundary_artifacts = aggregate.get('boundary_artifact_percentage', 0)
        continuous_edge_pct = aggregate.get('continuous_edge_percentage', 0)
        boundary_edges = aggregate.get('boundary_edges_detected', [])
        
        # Use either the clear signal OR percentage thresholds
        should_retry = requires_adjustment or (
            boundary_artifacts > boundary_artifact_threshold and boundary_edges
        ) or (
            continuous_edge_pct > 5  # Continuous edges are more serious
        )
        
        if should_retry:
            print("="*80)
            print("⚠️  BOUNDARY ARTIFACTS DETECTED - ADJUSTING BORDER DETECTION")
            print("="*80)
            
            if requires_adjustment:
                print("BRNG analyzer flagged that border adjustment is required")
            
            if continuous_edge_pct > 0:
                print(f"Found continuous edge artifacts in {continuous_edge_pct:.1f}% of frames")
            elif boundary_artifacts > 0:
                print(f"Found boundary artifacts in {boundary_artifacts:.1f}% of frames")
            
            if boundary_edges:
                print(f"Affected edges: {', '.join(set(boundary_edges))}")
            
            print("Re-running border detection with adjusted parameters...")
            
            # Calculate adjusted parameters based on which edges have artifacts
            # More aggressive adjustments for continuous edges
            if continuous_edge_pct > 5:
                # Strong adjustments for continuous edge artifacts
                adjusted_threshold = border_threshold + 10  # Much higher threshold
                adjusted_padding = border_padding + 10      # Much more padding
                adjusted_sample_frames = min(border_sample_frames + 30, 100)
                adjusted_edge_width = min(border_edge_sample_width + 100, 200)
            else:
                # Moderate adjustments for scattered edge artifacts
                adjusted_threshold = border_threshold + 5
                adjusted_padding = border_padding + 5
                adjusted_sample_frames = min(border_sample_frames + 20, 100)
                adjusted_edge_width = border_edge_sample_width + 50
            
            # Edge-specific adjustments
            if 'left' in boundary_edges:
                adjusted_edge_width = min(adjusted_edge_width + 30, 250)
            if 'right' in boundary_edges:
                adjusted_edge_width = min(adjusted_edge_width + 30, 250)
            if 'top' in boundary_edges:
                adjusted_padding += 3
            if 'bottom' in boundary_edges:
                adjusted_padding += 3
            
            print(f"  Adjusted parameters:")
            print(f"    Threshold: {border_threshold} → {adjusted_threshold}")
            print(f"    Edge sample width: {border_edge_sample_width} → {adjusted_edge_width}")
            print(f"    Sample frames: {border_sample_frames} → {adjusted_sample_frames}")
            print(f"    Padding: {border_padding} → {adjusted_padding}")
            
            # Re-run border detection
            print("\nRe-detecting borders with adjusted parameters...")
            border_results = detect_video_borders(
                video_path, 
                output_dir, 
                target_viz_time=viz_time,
                search_window=120,
                threshold=adjusted_threshold,
                edge_sample_width=adjusted_edge_width,
                sample_frames=adjusted_sample_frames,
                padding=adjusted_padding
            )
            
            border_retry_performed = True
            
            # Update border data path (it should be the same file, just updated)
            border_data_path = output_dir / f"{video_path.stem}_border_data.json"
            
            print()
            
            # Step 4: Re-run BRNG analysis with new borders
            print("STEP 4: RE-RUNNING BRNG ANALYSIS WITH ADJUSTED BORDERS")
            print("-" * 40)
            
            if border_results.get('active_area'):
                print("Re-analyzing BRNG violations with adjusted active area...")
                brng_results = analyze_active_area_brng(
                    video_path=video_path,
                    border_data_path=border_data_path,
                    output_dir=output_dir,
                    duration_limit=brng_duration
                )
                
                # Check if the retry was successful
                new_aggregate = brng_results.get('aggregate_patterns', {})
                new_boundary_artifacts = new_aggregate.get('boundary_artifact_percentage', 0)
                new_continuous_edges = new_aggregate.get('continuous_edge_percentage', 0)
                new_requires_adjustment = new_aggregate.get('requires_border_adjustment', False)
                
                if not new_requires_adjustment:
                    print(f"✔ Border adjustment successful!")
                    if continuous_edge_pct > 0:
                        print(f"  Continuous edge artifacts reduced from {continuous_edge_pct:.1f}% to {new_continuous_edges:.1f}%")
                    if boundary_artifacts > 0 and new_boundary_artifacts < boundary_artifacts:
                        print(f"  Boundary artifacts reduced from {boundary_artifacts:.1f}% to {new_boundary_artifacts:.1f}%")
                else:
                    print(f"⚠️ Some edge artifacts remain ({new_boundary_artifacts:.1f}%) - may need manual adjustment")
                    print("  Consider further increasing border detection parameters")
            else:
                print("Still no borders detected after adjustment")
                brng_results = analyze_active_area_brng(
                    video_path=video_path,
                    border_data_path=None,
                    output_dir=output_dir,
                    duration_limit=brng_duration
                )
    
    print()
    
    # Step 5: FFprobe Signalstats Analysis (with final borders)
    step_num = 5 if border_retry_performed else 3
    print(f"STEP {step_num}: FFPROBE SIGNALSTATS ANALYSIS")
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
        if border_retry_performed:
            print(f"   ⚠️ Borders adjusted after initial BRNG analysis")
            print(f"   ✔ Automatic border correction applied")
        print(f"   Active area: {w}x{h} ({active_percentage:.1f}% of frame)")
        print(f"   Borders: L={x}px, R={border_results['video_properties']['width']-x-w}px, T={y}px, B={border_results['video_properties']['height']-y-h}px")
        
        if border_results.get('head_switching_artifacts', {}).get('severity') != 'none':
            print(f"   ⚠️ Head switching artifacts: {border_results['head_switching_artifacts']['severity']}")
    else:
        print(f"\n1. Border Detection: No borders detected (full frame active)")
    
    # Active Area BRNG Analysis
    print(f"\n2. Active Area BRNG Analysis (improved detection, up to {brng_duration}s):")
    if brng_results:
        actionable_report = brng_results.get('actionable_report', {})
        analyzed_region = 'active area' if brng_results['video_info'].get('active_area') else 'full frame'
        
        # Note about skipped content
        content_start = brng_results.get('video_info', {}).get('content_start_time', 0)
        if content_start > 0:
            print(f"   Skipped test patterns: Started at {content_start:.1f}s")
        
        print(f"   Method: Differential detection with edge emphasis")
        print(f"   Region analyzed: {analyzed_region}")
        if border_retry_performed:
            print(f"   ✔ Using adjusted borders from retry")
        print(f"   Assessment: {actionable_report.get('overall_assessment', 'Analysis complete')}")
        print(f"   Priority: {actionable_report.get('action_priority', 'none').upper()}")
        
        if brng_results['frames_with_violations'] > 0:
            print(f"   Frames with violations: {brng_results['frames_with_violations']}/{brng_results['total_frames_analyzed']}")
            print(f"   Maximum violation: {brng_results['max_violation_percentage']:.4f}% of pixels")
            
            # Show if edge violations remain
            aggregate = brng_results.get('aggregate_patterns', {})
            if aggregate.get('edge_violation_percentage', 0) > 0:
                print(f"   Edge violations: {aggregate['edge_violation_percentage']:.1f}% of frames")
        else:
            print(f"   ✔ No BRNG violations detected")
        
        # Mention thumbnails if saved
        if brng_results.get('saved_thumbnails'):
            print(f"   📸 Saved {len(brng_results['saved_thumbnails'])} diagnostic thumbnail(s)")
    else:
        print(f"   ⚠️ Analysis not available")
    
    # FFprobe Signalstats Summary
    print(f"\n3. FFprobe Signalstats (sample: {start_time}s-{start_time+duration}s):")
    print(f"   {signalstats_results.get('diagnosis', 'Analysis completed')}")
    
    if signalstats_results.get('results', {}).get('active_area'):
        active_area = signalstats_results['results']['active_area']
        print(f"   Active area: {active_area['violation_percentage']:.1f}% frames, max {active_area['max_brng']:.4f}% pixels")
        
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
    
    # Cross-analysis insights
    print("\n" + "-" * 40)
    print("CROSS-ANALYSIS INSIGHTS:")
    
    if border_retry_performed:
        print("✔ Border detection was automatically adjusted based on BRNG analysis")
        print("   → This provides more accurate results for all analyses")
    
    # Check if any edge issues remain
    if brng_results:
        aggregate = brng_results.get('aggregate_patterns', {})
        if aggregate.get('requires_border_adjustment'):
            print("⚠️ Edge artifacts still detected - consider manual border adjustment")
            print(f"   Edges affected: {', '.join(aggregate.get('boundary_edges_detected', []))}")
    
    # Compare FFprobe and BRNG results
    ffprobe_data = None
    if signalstats_results.get('results', {}).get('active_area'):
        ffprobe_data = signalstats_results['results']['active_area']
    elif signalstats_results.get('results', {}).get('full_frame'):
        ffprobe_data = signalstats_results['results']['full_frame']
    
    if ffprobe_data and brng_results:
        ffprobe_max = ffprobe_data.get('max_brng', 0)
        detailed_max = brng_results.get('max_violation_percentage', 0)
        
        if ffprobe_max < 0.01 and detailed_max < 0.01:
            print("✔ Both analyses confirm minimal BRNG violations")
            print("   → Video appears broadcast-safe")
        elif ffprobe_max > 0.1 or detailed_max > 0.1:
            print("⚠️ Both analyses detect notable BRNG violations")
            
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
    
    print("\n" + "-" * 40)
    print(f"\nOutput files generated:")
    print(f"  - {video_path.stem}_border_detection.jpg (border visualization)")
    print(f"  - {video_path.stem}_border_data.json (border detection data)")
    print(f"  - {video_path.stem}_active_brng_analysis.json (enhanced BRNG analysis)")
    print(f"  - {video_path.stem}_signalstats_analysis.json (FFprobe analysis)")
    if brng_results and brng_results.get('saved_thumbnails'):
        print(f"  - brng_thumbnails/ ({len(brng_results['saved_thumbnails'])} diagnostic thumbnails)")
    
    # Save combined summary
    summary_path = output_dir / f"{video_path.stem}_combined_summary.json"
    
    signalstats_summary_data = {}
    if signalstats_results.get('results', {}).get('active_area'):
        signalstats_summary_data['active_area_violations'] = signalstats_results['results']['active_area'].get('violation_percentage')
        signalstats_summary_data['active_area_max_brng'] = signalstats_results['results']['active_area'].get('max_brng')
    if signalstats_results.get('results', {}).get('full_frame'):
        signalstats_summary_data['full_frame_violations'] = signalstats_results['results']['full_frame'].get('violation_percentage')
        signalstats_summary_data['full_frame_max_brng'] = signalstats_results['results']['full_frame'].get('max_brng')
    
    brng_summary_data = {}
    if brng_results:
        actionable = brng_results.get('actionable_report', {})
        brng_summary_data = {
            'analyzed': True,
            'method': 'differential_detection',
            'region': 'active area' if brng_results['video_info'].get('active_area') else 'full frame',
            'frames_analyzed': brng_results.get('total_frames_analyzed', 0),
            'frames_with_violations': brng_results.get('frames_with_violations', 0),
            'max_violation_percentage': brng_results.get('max_violation_percentage', 0),
            'assessment': actionable.get('overall_assessment', 'Unknown'),
            'priority': actionable.get('action_priority', 'none'),
            'issues_detected': [rec['issue'] for rec in actionable.get('recommendations', [])],
            'temporal_pattern': brng_results.get('temporal_analysis', {}).get('temporal_pattern', 'unknown') if brng_results.get('temporal_analysis') else None,
            'thumbnails_saved': len(brng_results.get('saved_thumbnails', [])),
            'border_retry_performed': border_retry_performed
        }
    else:
        brng_summary_data = {'analyzed': False}
    
    combined_results = {
        'video_file': str(video_path),
        'analyses_performed': {
            'border_detection': True,
            'border_retry': border_retry_performed,
            'active_area_brng_differential': brng_results is not None,
            'ffprobe_signalstats': True
        },
        'border_results_summary': {
            'has_borders': border_results.get('active_area') is not None,
            'active_area': border_results.get('active_area'),
            'head_switching': border_results.get('head_switching_artifacts', {}).get('severity', 'none'),
            'adjusted_after_brng': border_retry_performed
        },
        'active_brng_summary': brng_summary_data,
        'signalstats_summary': {
            'diagnosis': signalstats_results.get('diagnosis'),
            **signalstats_summary_data
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"  - {video_path.stem}_combined_summary.json (combined analysis summary)")
    
    return {
        'border_results': border_results,
        'signalstats_results': signalstats_results,
        'active_brng_results': brng_results,
        'border_retry_performed': border_retry_performed
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