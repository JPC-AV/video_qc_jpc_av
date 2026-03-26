# AV Spex

AV processing application for digital preservation

![AV Spex logo](https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/images_for_readme/av_spex_the_logo.png?raw=true)

AV Spex is a macOS application written in Python that helps process digital audio and video media created from analog sources. It confirms that digitized files conform to predetermined specifications and performs automated preservation actions: fixity checks, access file creation, metadata sidecars, and HTML reports.

Designed for audiovisual preservation workflows, AV Spex was developed in the context of the Johnson Publishing Company archive at the Smithsonian's National Museum of African American History and Culture (NMAAHC).

---

## Quick Start

### Install (Homebrew)
```bash
brew tap JPC-AV/AV-Spex
brew install av-spex
```

### Run on a directory
```bash
av-spex /path/to/video_files
```

### Launch the GUI
```bash
av-spex-gui
```

---

## Requirements

macOS 13 (Ventura) and up

### Required Command Line Tools

The following command line tools must be installed separately. The macOS package manager [Homebrew](https://brew.sh/) is recommended:

- **[ExifTool](https://exiftool.org/)** — embedded metadata extraction
- **[FFmpeg](https://www.ffmpeg.org/)** — stream analysis, access file creation, stream hashing
- **[MediaConch](https://mediaarea.net/MediaConch)** — policy-based conformance validation
- **[MediaInfo](https://mediaarea.net/en/MediaInfo)** — container and stream metadata extraction
- **[MKVToolNix](https://mkvtoolnix.download/)** - a set of tools to create, alter and inspect Matroska files 
- **[QCTools](https://bavc.org/programs/preservation/preservation-tools/)** — per-frame video quality analysis

Install with Homebrew:
```bash
brew install exiftool ffmpeg mediaconch mediainfo mkvtoolnix
```

QCTools must be installed separately from [MediaArea](https://mediaarea.net/QCTools).

The AV Spex GUI checks for all required dependencies at startup:

![Dependency Check](https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/images_for_readme/dependency_check_0102026.png?raw=true)

---

## Installation

There are three installation options:

### 1. DMG (Recommended)

Download the installer from the [latest release](https://github.com/JPC-AV/JPC_AV_videoQC/releases/latest).

### 2. Homebrew 

```bash
brew tap JPC-AV/AV-Spex
brew install av-spex
```

Verify the installation:
```bash
av-spex --help
```

### 3. From Source

Python 3.10 or higher is required.

<details>
<summary><span style="font-style: italic;">Click for instructions on creating a virtual environment (optional)</span></summary>

Creating a virtual environment is optional but recommended to avoid system-wide package conflicts.

**Using Conda:**

1. Install: `brew install --cask anaconda`
2. Add to PATH: `export PATH="/opt/homebrew/anaconda3/bin:$PATH"` (Apple Silicon) or `export PATH="/usr/local/anaconda3/bin:$PATH"` (Intel)
3. Initialize: `conda init zsh`
4. Create environment: `conda create -n JPC_AV python=3.10.13`

**Using venv:**

```bash
python3 -m venv name_of_env
source ./name_of_env/bin/activate
```

</details>

```bash
cd path-to/JPC_AV_videoQC
pip install .
av-spex --help
```

---

## GUI Usage

Launch the GUI:
```bash
av-spex-gui
```

The GUI has four tabs: **Import**, **Checks**, **Spex**, and **Complex**

### Import Tab

<img src="https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/images_for_readme/avspex_import_tab.png?raw=true" alt="AV Spex Import Tab" width="400"/>

The Import tab is where you select input directories for processing and manage configuration files. It includes options to import, export, or reset the Checks and Spex configurations as JSON files.

### Checks Tab

<img src="https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/images_for_readme/avspex_checks_tab.png?raw=true" alt="AV Spex Checks Tab" width="400"/>

The Checks tab controls which tools and processing steps are run. It includes:

- **Checks Profiles**: Apply a preset profile (Step 1, Step 2, or Off) that configures a predefined set of tool options.
  - **Step 1**: Run and check ExifTool, FFprobe, MediaInfo, MediaTrace, and MediaConch; embed and output fixity
  - **Step 2**: Run QCTools and qct-parse (bar detection, evaluate bars, thumbnail export); validate fixity; generate HTML report
  - **Off**: Turn off all tools
- **Checks Options**: Enable or disable individual tools and checks using checkboxes. Each tool has a **Run Tool** option (generates a sidecar file) and a **Check Tool** option (compares the sidecar output against expected Spex values).

Click **Check Spex!** to start processing.

### Spex Tab

The Spex tab displays the expected metadata values that AV Spex validates against, organized by tool. It includes:

- **Filename / Signal Flow**: Dropdown menus to select the active filename convention and signal flow equipment profiles
- **ExifTool / MediaInfo / FFprobe**: Dropdown menus to select named expected-values profiles for each metadata tool (see [Custom Metadata Profiles](#custom-metadata-profiles) below)
- **Open Section**: View the current expected values for any section (read-only for default profiles)

Once your Spex selections are complete, navigate to the Checks tab and click **Check Spex!**.

### Complex Tab

The Complex tab provides configuration for QCTools, qct-parse, and frame analysis — the more advanced processing steps that are typically run during Step 2 or configured independently.

- **QCTools**: Toggle QCTools analysis on or off and set the output file extension (`qctools.xml.gz` or `qctools.mkv`)
- **qct-parse**: Enable or disable qct-parse sub-steps including bars detection, bar evaluation, and thumbnail export
- **Frame Analysis**: Configure the three frame analysis sub-steps (see [Frame Analysis](#frame-analysis) below for details):
  - **Border Detection**: Toggle on/off, select mode (simple or sophisticated), and set pixel crop width
  - **BRNG Analysis**: Toggle on/off, set maximum analysis duration, and enable or disable automatic color bar skipping
  - **Signalstats**: Toggle on/off

---

## Custom Metadata Profiles

AV Spex supports named expected-value profiles for ExifTool, MediaInfo, and FFprobe. This is useful when processing collections with different technical specifications — for example, SD vs. HD transfers, or FLAC vs. PCM audio.

Each tool's section in the Spex tab includes:
- A **profile dropdown** to select from saved profiles
- **Create Custom Profile...** to define a new set of expected values
- **Edit Selected Profile...** to modify an existing custom profile

Default profiles are protected from accidental modification or deletion. Custom profiles are saved to the user config directory and persist across sessions.

Multiple acceptable values can be defined for any field. For example, if a collection includes both FLAC and PCM audio, the expected `codec_name` can be set to `["flac", "pcm_s24le"]`.

Profiles can also be applied from the CLI:
```bash
av-spex --exiftool-profile "My ExifTool Profile"
av-spex --mediainfo-profile "My MediaInfo Profile"
av-spex --ffprobe-profile "My FFprobe Profile"
```

To view available profiles and current expected values:
```bash
av-spex -pp exiftool
av-spex -pp mediainfo
av-spex -pp ffprobe
```

---

## Frame Analysis

AV Spex includes a frame analysis module for detecting common analog video artifacts. It has three independently togglable sub-steps, all controlled from the Checks config.

### Border Detection

Detects the active video area and identifies edge artifacts including head-switching noise at the bottom of the frame.

![Border Detection Example](https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/images_for_readme/JPC_AV_01709_border_detection.jpg?raw=true)

Two modes are available:
- **Simple** (default): Crops a fixed pixel border from each edge (default: 25px)
- **Sophisticated**: Uses edge detection to dynamically identify the active video area

### BRNG Analysis

Detects out-of-range luma and chroma values (BRNG — **B**roadcast **Ra**n**g**e) using a multi-method voting approach. Frames with violations are highlighted in the diagnostic output, and results are included in the HTML report. BRNG analysis automatically skips color bars at the head of the tape to avoid false positives.

![BRNG Analysis Example](https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/images_for_readme/avspex_brng_example.png?raw=true)

### Signalstats

Runs FFmpeg's `signalstats` filter over sampled time periods (default: 3 periods of 60 seconds each) to assess signal quality across the tape.

### Frame Analysis CLI Flags

```bash
av-spex --enable-border-detection {on,off}
av-spex --enable-brng-analysis {on,off}
av-spex --enable-signalstats {on,off}
av-spex --frame-borders {simple,sophisticated}
av-spex --frame-border-pixels 25
av-spex --frame-brng-duration 300
av-spex --frame-no-colorbar-skip
```

---

## CLI Usage

```bash
av-spex [path/to/directory]
```

### Options

**Processing profiles:**
- `--profile {step1,step2,off}` — Apply a predefined processing profile (see [Checks Tab](#checks-tab) for details on each profile)

**Tool toggles:**
- `--on / --off` — Enable or disable individual tool options without affecting others. Format: `tool.run_tool` or `tool.check_tool` (e.g., `--on mediainfo.run_tool --on mediainfo.check_tool`)

**Spex profiles:**
- `--signalflow / -sn` — Select a signal flow equipment profile by name
- `--filename / -fn` — Select a filename convention profile by name
- `--exiftool-profile` — Apply a named ExifTool expected-values profile
- `--mediainfo-profile` — Apply a named MediaInfo expected-values profile
- `--ffprobe-profile` — Apply a named FFprobe expected-values profile

**Config management:**
- `--printprofile / -pp` — Print current config values. Accepts: `all`, `spex`, `checks`, `checks,tools`, `spex,filename_values`, `exiftool`, `mediainfo`, `ffprobe`, `signalflow`
- `--export-config {all,spex,checks}` — Export current config(s) to JSON
- `--export-file FILENAME` — Specify output filename for `--export-config`
- `--import-config FILE` — Import config from a previously exported JSON file
- `--mediaconch-policy FILE` — Import a custom MediaConch XML policy file
- `--use-default-config` — Reset all configs to defaults

**Other:**
- `-dr / --dryrun` — Apply config changes without processing any video files
- `--gui` — Force launch in GUI mode

### Example Workflow

1. Digitize analog media to MKV
2. Run initial metadata extraction and fixity:
   ```bash
   av-spex -d ./digitized_files --profile step1
   ```
3. Run quality analysis and report generation:
   ```bash
   av-spex -d ./digitized_files --profile step2
   ```
4. Review the HTML report, fixity results, and metadata outputs
5. Address any issues found

---

## Configuration

AV Spex's settings are stored in two primary JSON config files, editable through the GUI or CLI.

### Checks Config

Controls which tools run and what outputs are generated.

**Outputs**
- `access_file` — Create a low-resolution MP4 access copy
- `report` — Generate an HTML summary report
- `qctools_ext` — Output extension for QCTools files (e.g., `qctools.xml.gz` or `qctools.mkv`)
- **Frame Analysis** settings: `enable_border_detection`, `enable_brng_analysis`, `enable_signalstats`, `border_detection_mode` (simple/sophisticated), `simple_border_pixels` (default: 25), `brng_duration_limit` (default: 300 seconds)

**Fixity**
- `output_fixity` — Write checksums to a fixity text file
- `check_fixity` — Validate against stored checksums
- `embed_stream_fixity` — Embed video/audio stream hashes into MKV tags
- `validate_stream_fixity` — Validate against embedded stream hashes
- `overwrite_stream_fixity` — Overwrite existing embedded hashes

**Tools** — Each tool has `run_tool` and/or `check_tool` toggles:
- `exiftool`, `ffprobe`, `mediainfo`, `mediatrace`: `run_tool` and `check_tool`
- `mediaconch`: `run_mediaconch` and `mediaconch_policy` (path to XML policy file)
- `qctools`: `run_tool`
- `qct_parse`: `run_tool`, `barsDetection`, `evaluateBars`, `thumbExport`

### Spex Config

Stores expected metadata values organized by tool. Multiple acceptable values are supported using a list:
```json
"codec_name": ["flac", "pcm_s24le"]
```

Sections include: `filename_values`, `mediainfo_values` (general, video, audio track values), `exiftool_values`, `ffmpeg_values` (video stream, audio stream, format values), `mediatrace_values` (custom MKV tags like `ENCODER_SETTINGS`), and `qct_parse_values` (color bar thresholds and content filter definitions).

### Managing Configs

To export or import configurations:
```bash
av-spex --export-config all --export-file my_config.json
av-spex --import-config my_config.json
```

Configs can also be imported, exported, and reset from the Import tab in the GUI.

---

## Outputs

For each processed input directory `{video_id}/`:

- **`{video_id}_qc_metadata/`** — Sidecar metadata files (ExifTool, FFprobe, MediaConch, MediaInfo, MediaTrace, QCTools), fixity files, and a per-file log
- **`{video_id}_report_csvs/`** — CSV files used to populate the HTML report
- **`{video_id}_avspex_report.html`** — HTML summary report
- **`{video_id}_vrecord_metadata/`** — Legacy vrecord metadata, moved here if present

### Logging

Each run writes a timestamped application log:
```
logs/YYYY-MM-DD_HH-MM-SS_JPC_AV_log.log
```

A per-file log is also written inside each `_qc_metadata` directory.

---

## Contributing

Contributions that enhance script functionality are welcome. Please ensure compatibility with Python 3.10 or higher.

---

## Acknowledgements

AV Spex makes use of code from several open source projects. Attribution and copyright notices are included as comments inline where open source code is used.

[loglog](https://github.com/amiaopensource/loglog)
```
Copyright (C) 2021  Eddy Colloton and Morgan Morel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as published by
    the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

[qct-parse](https://github.com/amiaopensource/qct-parse)
```
Copyright (C) 2016 Brendan Coates and Morgan Morel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as published by
    the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

[IFIscripts](https://github.com/kieranjol/IFIscripts)
```
MIT License

    Copyright (c) 2015-2018 Kieran O'Leary for the Irish Film Institute.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
```