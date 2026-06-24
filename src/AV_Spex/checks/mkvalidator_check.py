#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import csv

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_manager import ConfigManager

config_mgr = ConfigManager()

# mkvalidator prints a long run of '.' characters as a progress indicator. In the
# captured sidecar that run is glued to the start of the final verdict line, e.g.
# "...........mkvalidator 0.6.0: the file appears to be valid". We strip leading
# dots from every line before interpreting it.
_PROGRESS_DOTS = '.'

# Each non-incrementing-cluster warning looks like:
#   WRN0C2: The timecode of the Cluster at 710818 is not incrementing (may be intentional)
# The integer is a byte offset into the file (not a media timecode), so it is
# reported as-is with no NDF/DF conversion.
_WRN0C2_RE = re.compile(r'^WRN0C2:.*Cluster at (\d+)', re.IGNORECASE)
_CODE_RE = re.compile(r'^(WRN[0-9A-F]+|ERR[0-9A-F]+):\s*(.*)$', re.IGNORECASE)


def parse_mkvalidator(file_path):
    """
    Parse an mkvalidator text sidecar.

    mkvalidator is a Matroska conformance validator rather than a metadata
    extractor, so this does not compare against SpexConfig. Instead it:

      * strips the '....' progress noise,
      * collects WRN0C2 (non-incrementing cluster timecode) warnings as a list of
        cluster byte offsets, written to a CSV for a future HTML report section,
      * writes a cleaned text summary sidecar (verdict + track/file lines), and
      * reports a difference only on hard errors (ERR* lines) or an "invalid"
        verdict. WRN0C2 and other warnings are informational and never fail.

    Returns a differences dict ``{field: [actual, expected]}`` on failure, else None.
    """
    config_mgr.refresh_configs()

    if not os.path.exists(file_path):
        logger.critical(f"Cannot perform mkvalidator check! No such file: {file_path}")
        return None

    try:
        with open(file_path, 'r', errors='replace') as f:
            raw_lines = f.readlines()
    except IOError as e:
        logger.error(f"Failed to read mkvalidator output {file_path}: {e}")
        return None

    cluster_offsets = []       # WRN0C2 byte offsets (the future report section)
    other_warnings = []        # (code, message) for non-WRN0C2 warnings
    errors = []                # (code, message) for ERR* lines
    summary_lines = []         # cleaned non-warning/non-error lines (verdict, tracks, file)
    verdict = None

    for raw in raw_lines:
        line = raw.lstrip(_PROGRESS_DOTS).strip()
        if not line:
            continue

        wrn0c2 = _WRN0C2_RE.match(line)
        if wrn0c2:
            cluster_offsets.append(int(wrn0c2.group(1)))
            continue

        code_match = _CODE_RE.match(line)
        if code_match:
            code, message = code_match.group(1).upper(), code_match.group(2).strip()
            if code.startswith('ERR'):
                errors.append((code, message))
            else:
                other_warnings.append((code, message))
            continue

        # Non-coded line: verdict / track info / file path / "created with".
        summary_lines.append(line)
        if 'appears to be valid' in line.lower() or 'is not valid' in line.lower() \
                or 'appears to be invalid' in line.lower():
            verdict = line

    verdict_invalid = verdict is not None and (
        'invalid' in verdict.lower() or 'not valid' in verdict.lower()
    )
    is_invalid = bool(errors) or verdict_invalid

    _write_summary_sidecar(file_path, verdict, errors, other_warnings,
                           cluster_offsets, summary_lines)
    _write_cluster_csv(file_path, cluster_offsets)

    # Log a concise result line.
    if cluster_offsets:
        logger.info(f"mkvalidator: {len(cluster_offsets)} WRN0C2 non-incrementing "
                    f"cluster timecode warning(s) (informational).")
    if not is_invalid:
        logger.info(f"mkvalidator: {verdict or 'the file appears to be valid'}\n")
        return None

    logger.critical("mkvalidator reported errors or an invalid file:")
    differences = {}
    if verdict:
        differences['mkvalidator validity'] = [verdict, 'the file appears to be valid']
    for code, message in errors:
        logger.critical(f"  {code}: {message}")
        differences[f'mkvalidator {code}'] = [message, 'no error reported']
    if not differences:
        # ERR present but no parseable detail — still surface a failure row.
        differences['mkvalidator validity'] = ['mkvalidator reported errors',
                                               'the file appears to be valid']
    logger.debug("")
    return differences


def _strip_video_id(file_path):
    """Recover the {video_id} prefix from a {video_id}_mkvalidator_output.txt path."""
    base = os.path.basename(file_path)
    return re.sub(r'_mkvalidator_output\.[^.]+$', '', base)


def _write_summary_sidecar(file_path, verdict, errors, other_warnings,
                           cluster_offsets, summary_lines):
    """
    Write a cleaned text summary (progress dots removed) next to the raw output.
    """
    out_dir = os.path.dirname(file_path)
    video_id = _strip_video_id(file_path)
    summary_path = os.path.join(out_dir, f'{video_id}_mkvalidator_summary.txt')

    try:
        with open(summary_path, 'w') as out:
            out.write(f"{verdict or 'mkvalidator verdict not found'}\n\n")
            out.write(f"Errors (ERR): {len(errors)}\n")
            out.write(f"WRN0C2 non-incrementing cluster timecode warnings: "
                      f"{len(cluster_offsets)}\n")
            out.write(f"Other warnings: {len(other_warnings)}\n")

            if errors:
                out.write("\nErrors:\n")
                for code, message in errors:
                    out.write(f"  {code}: {message}\n")
            if other_warnings:
                out.write("\nOther warnings:\n")
                for code, message in other_warnings:
                    out.write(f"  {code}: {message}\n")

            # Remaining summary detail (track info, file path, created-with), with
            # the verdict de-duplicated since it is already at the top.
            detail = [ln for ln in summary_lines if ln != verdict]
            if detail:
                out.write("\nDetails:\n")
                for ln in detail:
                    out.write(f"  {ln}\n")
        logger.debug(f"mkvalidator summary written: {summary_path}")
    except IOError as e:
        logger.error(f"Failed to write mkvalidator summary {summary_path}: {e}")


def _write_cluster_csv(file_path, cluster_offsets):
    """
    Persist the WRN0C2 cluster byte offsets as a CSV for a future report section.

    Written alongside the raw output in the _qc_metadata directory because the
    _report_csvs directory does not exist yet when the metadata checks run.
    """
    if not cluster_offsets:
        return

    out_dir = os.path.dirname(file_path)
    video_id = _strip_video_id(file_path)
    csv_path = os.path.join(out_dir, f'{video_id}_mkvalidator_clusters.csv')

    try:
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['cluster_index', 'byte_offset'])
            for index, offset in enumerate(cluster_offsets, start=1):
                writer.writerow([index, offset])
        logger.debug(f"mkvalidator cluster CSV written: {csv_path}")
    except IOError as e:
        logger.error(f"Failed to write mkvalidator cluster CSV {csv_path}: {e}")
