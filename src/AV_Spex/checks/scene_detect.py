"""
Scene-cut detection via PySceneDetect.

Stage 1: produces a scene-boundaries CSV next to the other qc_metadata
sidecars. The boundaries are emitted observation-only — Stage 2 will pass
this list to frame_analysis (BRNG, signalstats, duplicate-frame detection)
and qct-parse audio analysis to suppress false positives that coincide with
cuts in the source material.

Detector choice (config: tools.scene_detection.detector):
    - 'content'  : ContentDetector — HSV-based, the standard cut detector
    - 'adaptive' : AdaptiveDetector — rolling-window thresholding, more
                   robust on noisy analog footage where a fixed threshold
                   over-fires on grain or head-switching artifacts

PySceneDetect is a Python package (not a CLI tool), so it is imported lazily
inside run_scene_detection() — that way an environment that doesn't ship
the dep can still import this module without crashing.
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from AV_Spex.utils.log_setup import logger


DURATIONS_CSV_SUFFIX = "_scene_boundaries.csv"


@dataclass
class SceneBoundary:
    """A single scene cut.

    A boundary is the *first frame of a new scene*: the cut occurs between
    (frame_num - 1) and frame_num. Stage 2 filtering treats a window of
    frame_padding frames on either side as "near a cut".
    """
    scene_index: int          # 1-based index of the scene this boundary starts
    frame_num: int            # first frame of the new scene (cut frame)
    timestamp_seconds: float  # presentation time of frame_num
    timecode: str             # HH:MM:SS.sss


def _format_timecode(seconds: float) -> str:
    """Format seconds as HH:MM:SS.sss to match qct-parse / clams CSVs."""
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def _csv_path(report_directory: str, video_id: str) -> str:
    return os.path.join(report_directory, f"{video_id}{DURATIONS_CSV_SUFFIX}")


def _write_csv(report_directory: str, video_id: str,
               boundaries: List[SceneBoundary]) -> str:
    """Write the scene-boundaries CSV. Always written, even when empty."""
    Path(report_directory).mkdir(parents=True, exist_ok=True)
    out_path = _csv_path(report_directory, video_id)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scene_index", "frame_num", "timestamp_seconds", "timecode"])
        for b in boundaries:
            writer.writerow([b.scene_index, b.frame_num,
                             f"{b.timestamp_seconds:.6f}", b.timecode])
    return out_path


def run_scene_detection(
    video_path: str,
    report_directory: str,
    video_id: str,
    detector: str = "content",
    threshold: float = 27.0,
    min_scene_len: int = 15,
    check_cancelled=None,
    signals=None,
) -> Optional[List[SceneBoundary]]:
    """Detect scene cuts and write the boundaries CSV.

    Returns the list of SceneBoundary objects (possibly empty), or None if
    PySceneDetect is unavailable / detection fails. The caller is expected
    to log this and continue — scene detection is observation-only.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector, AdaptiveDetector
    except ImportError:
        logger.warning(
            "Scene detection skipped: PySceneDetect is not installed. "
            "Install with: pip install scenedetect"
        )
        return None

    if check_cancelled and check_cancelled():
        return None

    logger.info(
        f"Scene detection: starting (detector={detector}, "
        f"threshold={threshold}, min_scene_len={min_scene_len})"
    )

    detector_lc = (detector or "content").lower()
    if detector_lc == "adaptive":
        # AdaptiveDetector uses adaptive_threshold, not threshold; min_scene_len
        # is shared. PySceneDetect's defaults for the others are sensible.
        scene_detector = AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=min_scene_len,
        )
    else:
        if detector_lc != "content":
            logger.warning(
                f"Unknown scene detector '{detector}', falling back to 'content'"
            )
        scene_detector = ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
        )

    if signals and hasattr(signals, "scene_detection_progress"):
        signals.scene_detection_progress.emit(0)

    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(scene_detector)
        scene_manager.detect_scenes(
            video=video,
            show_progress=False,
        )
        # scene_list is a list of (FrameTimecode start, FrameTimecode end);
        # the cut between scene N and N+1 lives at scene_list[N+1].start.
        scene_list = scene_manager.get_scene_list()
    except Exception as exc:  # noqa: BLE001 — surface anything as a soft failure
        logger.warning(f"Scene detection failed: {exc}")
        _write_csv(report_directory, video_id, [])
        return None

    if check_cancelled and check_cancelled():
        return None

    boundaries: List[SceneBoundary] = []
    # Skip scene_list[0]: its start is frame 0, which isn't a cut. Every
    # subsequent scene's start IS a cut between the previous scene and this one.
    for idx, (scene_start, _scene_end) in enumerate(scene_list):
        if idx == 0:
            continue
        frame_num = scene_start.get_frames()
        timestamp = scene_start.get_seconds()
        boundaries.append(SceneBoundary(
            scene_index=idx + 1,
            frame_num=frame_num,
            timestamp_seconds=timestamp,
            timecode=_format_timecode(timestamp),
        ))

    out_path = _write_csv(report_directory, video_id, boundaries)
    logger.info(
        f"Scene detection: finished — {len(boundaries)} cut(s) detected, "
        f"results written to {os.path.basename(out_path)}"
    )

    if signals and hasattr(signals, "scene_detection_progress"):
        signals.scene_detection_progress.emit(100)

    return boundaries
