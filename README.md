# JPC_AV
AV processing application for the Johnson Publishing Company archive

## Introduction:
![Alt text](https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/src/AV_Spex/logo_image_files/av_spex_the_logo.png?raw=true)
AV Spex is a macOS application written in python. The app is designed to help process digital audio and video media created from analog sources, by confirming that the digital files conform to predetermined specifications, and performing automated preservation actions. AV Spex performs fixity checks, creates access files, metadata sidecars, and html reports 

----

## Requirements

macOS 13 (Ventura) and up

### Required Command Line Tools

The following command line tools are necessary and must be installed separately. The macOS package manager ([homebrew](https://brew.sh/)) is recommended to install the following software:    

  - **[Exiftool](https://exiftool.org/)**
  - **[FFmpeg](https://www.ffmpeg.org/)**
  - **[MediaConch](https://mediaarea.net/MediaConch)**
  - **[MediaInfo](https://mediaarea.net/en/MediaInfo)**
  - **[QCTools](https://bavc.org/programs/preservation/preservation-tools/)**

If using homebrew, install each tool using:
`brew install [name of tool]`

## Installation of AV Spex

There are currently 3 installation options.
1. Install AV Spex using the provided DMG file: https://github.com/JPC-AV/JPC_AV_videoQC/releases/latest
2. Install AV Spex using the macOS package manger [homebrew](https://brew.sh/)
3. Download this code from github and install the Python version using `pip install .`

### 1. DMG Install

Download the DMG here: https://github.com/JPC-AV/JPC_AV_videoQC/releases/latest

### 2. Homebrew Install

For instructions on installing homebrew, see: https://brew.sh/

To install av-spex with homebrew first tap the formula's git repo:
```
brew tap JPC-AV/AV-Spex
```
Then install the app
```
brew install av-spex
```
Verify the installation by running:
```bash
av-spex --help
```

### 3. Python Install (from source)
Python 3.10 or higher is required.

---

- **Recommended - Create a Virtual Environment**   
   Best practices recommend python virtual environments to cleanly separate project dependencies, avoiding system-wide package conflicts.    
   Creating a virtual environment is optional - any Python 3.10+ environment should be compatible.    

<details>
<summary><span style="font-style: italic;">Click to expand instructions for creating a virtual environment</span></summary>

   - **Using Conda**    
      Conda is a cross-platform package and environment manager that creates isolated workspaces to simplify switching between project environments from the command line.

      1. **Install Conda:**
         - Via Homebrew: `brew install --cask anaconda`
         - Alternatively, follow the installation guide on [Anaconda's official website](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html).

      2. **Add Conda to Your Path:**
         - Installation paths may vary based on your system's architecture (x86 or ARM).
         - For Homebrew installations:
         - ARM architecture: `export PATH="/opt/homebrew/anaconda3/bin:$PATH"`
            - for Apple silicon 
         - x86 architecture: `export PATH="/usr/local/anaconda3/bin:$PATH"`
            - for Intel Mac 
         - If you are unsure which of these paths to use, you can check by running `brew --prefix`

      3. **Initialize Conda:**
         - For Bash: `conda init`
         - For Zsh: `conda init zsh`
         - To check your shell, run: `echo $SHELL`
      4. **Create an Isolated Environment**
         - To create an environment with the required Python version:
         ```bash
         conda create -n JPC_AV python=3.10.13
         ```
   - **Using venv**    
      Alternatively, use the built in virtual environment manager for your OS:

      1. **Create the environment**    
         Unix based (Mac or Linux):    
         `python3 -m venv name_of_env`     
         Windows:     
         `py -m venv name_of_env (where 'name_of_env' is replaced with the name of your virtual environment)`     
      2. **Activate virtual env**     
         Unix based (Mac or Linux):     
         `source ./name_of_env/bin/activate`     
         Windows:    
         `name_of_env\scripts\activate`     
</details>

---

### 3.1 Navigate to the Project Root Directory
   ```bash
   cd path-to/JPC_AV/JPC_AV_videoQC
   ```

### 3.2 Install AV Spex from source
   ```bash
   pip install .
   ```

### 3.3 Verify the installation by running
   ```bash
   av-spex --help
   ```

## GUI Usage
<img src="https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/src/AV_Spex/logo_image_files/germfree_eq.png" alt="graphic eq image" style="width:200px;"/>

To open the AV Spex gui from the command line application:
```bash
av-spex-gui
```

The GUI is divided into 2 tabs  - "Checks" and "Spex".   

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/images_for_readme/avspex_gui_screenshot_1.png" alt="AV Spex GUI Screenshot 1" width="400"/>
  <img src="https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/images_for_readme/avspex_gui_screenshot_2.png" alt="AV Spex GUI Screenshot 2" width="400"/>
</div>

### Checks
The "Checks" window displays the tools and commands that will be run on the imported directories.

- <b>Import Directories...</b>    
To import directories simply click the "Import Directory..." button in the GUI window or choose it from the "File" menu.    
- <b>Checks Profiles</b>    
Apply a pre-determined "profile" that applies a set of checks options via this dropdown menu.    
- <b>Checks Options</b>    
Edit the tool selections directly using the check boxes in the *checks options* window.    
- <b>Check Spex button</b>    
If you are ready to run the checks, click the "Check Spex!" button  

### Spex
The "Spex" section displays the expected values that AV Spex will be checking imported directories against. 

- <b>Expected Values Sections</b>    
Th expected values from are drawn popular metadata tools like exiftool, MeidaInfo, and FFprobe, as well as NMAAHC specific needs like file naming profiles and embedded signal flow documentation.      
- <b>Open Section</b>     
To view any of these specifications, click the "Open Section" button.       
The expected specifications cannot be edited from the "Open Section" text box window, those are for review only.     
- <b>Spex Dropdown Menus</b>    
To change the expected values of the file naming convention or the embedded signal flow documentation (checked by the MediaTrace tool), use the provided dropdown menus.

Once you have completed your Spex selections, navigate back to the Checks window to run the app using the "Check Spex!" button.


## CLI Usage
<img src="https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/src/AV_Spex/logo_image_files/germfree_eq.png" alt="graphic eq image" style="width:200px;"/>

Execute the scripts with:
```bash
av-spex [path/to/directory]
```

### av-spex --help
```bash
    // | |       ||   / /       //   ) )                              
   //__| |       ||  / /       ((            ___        ___           
  / ___  | ____  || / /          \\        //   ) )   //___) ) \\ / / 
 //    | |       ||/ /             ) )    //___/ /   //         \/ /  
//     | |       |  /       ((___ / /    //         ((____      / /\  

usage: av-spex [-h] [--version] [-dr] [--profile {step1,step2,off}]
               [--on {tool_name.run_tool, tool_name.check_tool}]
               [--off {tool_name.run_tool, tool_name.check_tool}]
               [-sn {JPC_AV_SVHS,BVH3100}] [-fn {jpc,bowser}]
               [-pp [PRINTPROFILE]] [-d] [-f] [--gui] [--use-default-config]
               [--export-config {all,spex,checks}] [--export-file EXPORT_FILE]
               [--import-config IMPORT_CONFIG]
               [--mediaconch-policy MEDIACONCH_POLICY]
               [paths ...]

av-spex 0.7.7

AV Spex is a python application designed to help process digital audio and video media created from analog sources.
The scripts will confirm that the digital files conform to predetermined specifications.

positional arguments:
  paths                 Path to the input -f: video file(s) or -d:
                        directory(ies)

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -dr, --dryrun         Flag to run av-spex w/out outputs or checks. Use to
                        change config profiles w/out processing video.
  --profile {step1,step2,off}
                        Select processing profile or turn checks off
  --on {tool_name.run_tool, tool_name.check_tool}
                        Turns on specific tool run_ or check_ option (format:
                        tool.check_tool or tool.run_tool, e.g.
                        mediainfo.run_tool)
  --off {tool_name.run_tool, tool_name.check_tool}
                        Turns off specific tool run_ or check_ option (format:
                        tool.check_tool or tool.run_tool, e.g.
                        mediainfo.run_tool)
  -sn {JPC_AV_SVHS,BVH3100}, --signalflow {JPC_AV_SVHS,BVH3100}
                        Select signal flow config type (JPC_AV_SVHS or
                        BVH3100)
  -fn {jpc,bowser}, --filename {jpc,bowser}
                        Select file name config type (jpc or bowser)
  -pp [PRINTPROFILE], --printprofile [PRINTPROFILE]
                        Show config profile(s) and optional subsection.
                        Format: 'config[,subsection]'. Examples: 'all',
                        'spex', 'checks', 'checks,tools',
                        'spex,filename_values'
  -d, --directory       Flag to indicate input is a directory
  -f, --file            Flag to indicate input is a video file
  --gui                 Force launch in GUI mode
  --use-default-config  Reset to default config by removing any saved
                        configurations
  --export-config {all,spex,checks}
                        Export current config(s) to JSON
  --export-file EXPORT_FILE
                        Specify export filename (default: auto-generated)
  --import-config IMPORT_CONFIG
                        Import configs from JSON file
  --mediaconch-policy MEDIACONCH_POLICY
                        Path to custom MediaConch policy XML file
```

<a name="options"></a> Options explained in detail [below](#options). 

#### Options    
The command line options can be used to edit the configurable settings described above.   

- `--profile`: Selects a predefined processing profile of particular tools outputs and checks    
   - Options: `step1`, `step2`, `off` 
- `--on`: Enables the specified tool without affecting others. Use the suffix ".run_tool" to run the specified tool, or ".check_tool" to check the output.
   - List multiple tools in this format: `--on exiftool.run_tool --on exiftool.check_tool --on mediainfo.run_tool --on mediainfo.check_tool --on ffprobe.run_tool --on ffprobe.check_tool`
- `--off`: Disables the specified tool without affecting others.
   - List multiple tools in this format: `--off exiftool.run_tool --off exiftool.check_tool --off mediainfo.run_tool --off mediainfo.check_tool --off ffprobe.run_tool --off ffprobe.check_tool`
- `--signalflow/-sn`: Changes the expected values in the config.yaml file for the mkv tag `ENCODER_SETTINGS` according to NMAAHC custom metadata convention   
   - Options: `JPC_AV_SVHS`, `BVH3100`
- `--filename/-fn`: Changes the expected values in the config.yaml for the input file naming convention
   - Options: `jpc`, `bowser`
- `--printprofile/-pp`: Prints the Checks and/or Spex profile. Print all Spex and Check with simply `-pp`, or specify a config, or config's subsection.
   - Options: `'config[,subsection]'`. Examples: `'all', 'spex', 'checks', 'checks,tools', 'spex,filename_values'`
- `--use-default-config`: Reset to default config by removing the last used reference file.
- `--export-config`: Export current config(s) to JSON. Default output filename is: `av_spex_config_export_YYYYMMDD_HHmmSS.json`. Requires one of 3 options:
   - Options: `{all,spex,checks}`
- `--export-file`: Must be used in combination with `--export-config`. Allows you to specify the exported config json file name and file path.
   - Example usage: `av-spex --export-config checks --export-config checks_config_output.json`
- `--import-config`: Import configs from JSON file. Can be used with json files exported using the `--export-config` and `--export-file` options described above.
- `--mediaconch-policy`: Import new mediaconch XML policy file and use this as the new policy. Once imported, the policy file will be available in the av-spex GUI.

<br/><br/>

<img src="https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/src/AV_Spex/logo_image_files/germfree_eq.png" alt="graphic eq image" style="width:200px;"/>
---

## Outputs
- Outputs are saved in a subdirectory within the input directory named: 
   - **[input_directory_name]_qc_metadata**:  Metadata outputs for: fixity check, exiftool, ffprobe, mediaconch, mediainfo, mediatrace, and qctools
   - **[input_directory_name]_report_csvs**: CSV files used to populate the HTML report summarizing the outputs
- An HTML file is output which collects the various outputs of AV Spex and presents them as a report named: [input_directory_name]_avspex_report.html
- Any existing vrecord metadata is moved to a subdirectory named: [input_directory_name]_vrecord_metadata    

## Logging
Each time AV Spex is run a log file is created. Everything output to the terminal is also recorded in a log file w/ timestamps located at:
```
logs/YYYY-MM-DD_HH-MM-SS_JPC_AV_log.log
```

## Config Files
The GUI and CLI options both edit an underlying set of json config files. To maintain consistent formatting the Checks and Spex config settings can only be edited using the GUI or the command line [options](#options).   
To customize the settings, you can export and edit the json files using the `--export-config/--import-config` options, but be careful to match the formatting exactly. 
The sections of the Checks and Spex configs are listed below.

### Checks Config:
The Checks Config stores settings pertaining to which output, tools and checks will be run.   
Each tool has a 'run' or 'check' option. **'run'** outputs a sidecar file. **'check'** compares the values in the sidecar file to the values stored in the Spex Config.     

- **Outputs**
   - **access_file**: yes/no
   - **report**: yes/no
   - **qctools_ext**: Can be any text string, for example `qctools.xml.gz` or `mkv.qctools.mkv`

- **Fixity**
   - **output_fixity**: yes/no
      - Generate and write md5 checksum to [input_video_file_name]_YYY_MM_DD_fixity.txt file
   - **check_fixity**: yes/no
      - Read md5 checksums from text files in the input directory that end with '_checksums.md5' or '_fixity.txt' and validate against calculated md5. Record result to [input_video_file_name]_YYY_MM_DD_fixity_check.txt   

   - **embed_stream_fixity**: yes/no    
      - Calculate video stream and audio stream md5 checksums using the ffmpeg command: `ffmpeg -loglevel error -i {input_video} -map 0 -f streamhash -hash md5 - `    
      The resulting stream hash is then embedded into the MKV file under the tag `VIDEO_STREAM_HASH` and `AUDIO_STREAM_HASH` using a temporary XML file and the mkvpropedit command: `mkvpropedit --tags "global:{temp_xml_file}" "{mkv_file}`    

   - **overwrite_stream_fixity**: yes/no
   - **validate_stream_fixity**: yes/no
      - Read existing audio and video 'streamhash' md5s found embedded in the input mkv video file with the tags `VIDEO_STREAM_HASH` or `AUDIO_STREAM_HASH` and validate against calculated md5

- **Tools**
   - **Exiftool**
      - **check_tool**: yes/no
      - **run_tool**: yes/no
   - **FFprobe**
      - **check_tool**: yes/no
      - **run_tool**: yes/no
   - **Mediaconch**
      - **mediaconch_policy**: mediaconch xml file   
      - **run_mediaconch**: yes/no
   - **Mediainfo**
      - **check_tool**: yes/no
      - **run_tool**: yes/no
   - **Mediatrace** (checks custom mkv tags)
      - **check_tool**: yes/no
      - **run_tool**: yes/no
   - **QCTools**
      - **run_tool**: yes/no
   - **QCT Parse** 
      - **run_tool**: yes/no
      - **barsDetection**: true/false
         - Find color bars, if present, and output start and end timestamp
      - **evaluateBars**: true/false
         - Identify maximum and minimum values for Y, Cb, Cr and Saturation in color bars. Using these maximums and minimums as thresholds, evaluate the rest of the video for values outside these values.
      - **contentFilter**: [Name of any content filter defined in the Spex Config]
         - Identify specific content types by their QCTools report values. For example, segments fo all black. 
      - **profile**: [Name of any threshold "profile" defined in the Spex Config]
         - Evaluate QCTools report values against a set of thresholds (called a 'profile'). Returns the percentage of frames outside of those thresholds per tag.
      - **tagname**: Set ad hoc thresholds per tag, using the following format: ` - [YMIN, lt, 100] `
      - **thumbExport**: true/false
         - Export thumbnail png image files for frames outside of set thresholds, limit is currently set as 1 thumbnail maximum for every 5 minutes of input video duration 
           
### Spex Config:
The Spex Config stores expected metadata values. The Checks compare the input against the expected values. As with the Checks config, the Spex are organized by tool.    
    
Each section stores metadata fields that correspond with output for that particular tool. Multiple acceptable values are allowed for all fields, wrapped in brackets like this: 
`"codec_name": ["flac", "pcm_s24le"]`

As with the Checks config, to maintain consistent formatting the Checks and Spex config settings can only be edited using the GUI or the command line [options](#options). 

- **filename_values**
  - Custom sections for filename profiles, toggled using CLI options `--filename/-fn` or from the dropdown in the Spex tab of the GUI
- **mediainfo_values**
  - **expected_general**
    - **file_extension**: "mkv"
    - **format**: "Matroska"
    - etc.
  - **expected_video**
    - **format**: "FFV1"
    - **format_settings_gop**: "N=1"
    - etc.
  - **expected_audio**
    - **format**: ["FLAC", "PCM"]
    - **channels**: "2 channels"
    - etc.
- **exiftool_values**
  - **file_type**: "MKV"
  - **file_type_extension**: "mkv"
  - etc.
- **ffmpeg_values**
  - **video_stream**
    - **codec_name**: "ffv1"
    - **codec_long_name**: "FFmpeg video codec #1"
    - etc.
  - **audio_stream**
    - **codec_name**: ["flac", "pcm_s24le"]
    - **codec_long_name**: ["FLAC (Free Lossless Audio Codec)", "PCM signed 24-bit little-endian"]
    - etc.
  - **format**
    - **format_name**: "matroska webm"
    - **format_long_name**: "Matroska / WebM"
    - etc.
- **mediatrace_values**
  - Used to check custom embedded mkv tags.
- **qct_parse_values**
  - etc.

## Contributing
Contributions that enhance script functionality are welcome. Please ensure compatibility with Python 3.10 or higher.

## Acknowledgements 
AV Spex makes use of code from several open source projects. Attribution and copyright notices are included as comments inline where open source code is used.    
The copyright notices are reproduced here:

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
