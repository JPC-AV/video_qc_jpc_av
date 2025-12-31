from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional
from enum import Enum

@dataclass
class FilenameSection:
    value: str
    section_type: str = field(default="literal") 

@dataclass
class FilenameValues:
    fn_sections: Dict[str, FilenameSection]
    FileExtension: str

@dataclass
class MediainfoGeneralValues:
    FileExtension: str
    Format: str
    OverallBitRate_Mode: str

@dataclass
class MediainfoVideoValues:
    Format: str
    Format_Settings_GOP: str
    CodecID: str
    Width: str
    Height: str
    PixelAspectRatio: str
    DisplayAspectRatio: str
    FrameRate_Mode_String: str
    FrameRate: str
    Standard: str
    ColorSpace: str
    ChromaSubsampling: str
    BitDepth: str
    ScanType: str
    ScanOrder: str
    Compression_Mode: str
    colour_primaries: str
    colour_primaries_Source: str
    transfer_characteristics: str
    transfer_characteristics_Source: str
    matrix_coefficients: str
    MaxSlicesCount: str
    ErrorDetectionType: str

@dataclass
class MediainfoAudioValues:
    Format: List[str]
    Channels: str
    SamplingRate: str
    BitDepth: str
    Compression_Mode: str

@dataclass
class ExiftoolValues:
    FileType: str
    FileTypeExtension: str
    MIMEType: str
    VideoFrameRate: str
    ImageWidth: str
    ImageHeight: str
    VideoScanType: str
    DisplayWidth: str
    DisplayHeight: str
    DisplayUnit: str
    CodecID: List[str]
    AudioChannels: str
    AudioSampleRate: str
    AudioBitsPerSample: str

@dataclass
class FFmpegVideoStream:
    codec_name: str
    codec_long_name: str
    codec_type: str
    codec_tag_string: str
    codec_tag: str
    width: str
    height: str
    display_aspect_ratio: str
    pix_fmt: str
    color_space: str
    color_transfer: str
    color_primaries: str
    field_order: str
    bits_per_raw_sample: str

@dataclass
class FFmpegAudioStream:
    codec_name: List[str]
    codec_long_name: List[str]
    codec_type: str
    codec_tag: str
    sample_fmt: str
    sample_rate: str
    channels: str
    channel_layout: str
    bits_per_raw_sample: str

@dataclass
class FFmpegFormat:
    format_name: str
    format_long_name: str
    tags: Dict[str, Optional[str]] = field(default_factory=lambda: {
        'creation_time': None,
        'ENCODER': None,
        'TITLE': None,
        'ENCODER_SETTINGS': EncoderSettings,
        'DESCRIPTION': None,
        'ORIGINAL MEDIA TYPE': None,
        'ENCODED_BY': None
    })

@dataclass
class EncoderSettings:
    Source_VTR: List[str] = field(default_factory=list)
    TBC_Framesync: List[str] = field(default_factory=list)
    ADC: List[str] = field(default_factory=list)
    Capture_Device: List[str] = field(default_factory=list)
    Computer: List[str] = field(default_factory=list)

@dataclass
class MediaTraceValues:
    COLLECTION: Optional[str]
    TITLE: Optional[str]
    CATALOG_NUMBER: Optional[str]
    DESCRIPTION: Optional[str]
    DATE_DIGITIZED: Optional[str]
    ENCODER_SETTINGS: EncoderSettings
    ENCODED_BY: Optional[str]
    ORIGINAL_MEDIA_TYPE: Optional[str]
    DATE_TAGGED: Optional[str]
    TERMS_OF_USE: Optional[str]
    _TECHNICAL_NOTES: Optional[str]
    _ORIGINAL_FPS: Optional[str]

@dataclass
class AllBlackContent:
    YMAX: tuple[float, str]
    YHIGH: tuple[float, str]
    YLOW: tuple[float, str]
    YMIN: tuple[float, str]

@dataclass
class StaticContent:
    YMIN: tuple[float, str]
    YLOW: tuple[float, str]
    YAVG: tuple[float, str]
    YMAX: tuple[float, str]
    YDIF: tuple[float, str]
    ULOW: tuple[float, str]
    UAVG: tuple[float, str]
    UHIGH: tuple[float, str]
    UMAX: tuple[float, str]
    UDIF: tuple[float, str]
    VMIN: tuple[float, str]
    VLOW: tuple[float, str]
    VAVG: tuple[float, str]
    VHIGH: tuple[float, str]
    VMAX: tuple[float, str]
    VDIF: tuple[float, str]

@dataclass
class Content:
    allBlack: AllBlackContent
    static: StaticContent

@dataclass
class FullTagList:
    YMIN: Optional[float]
    YLOW: Optional[float]
    YAVG: Optional[float]
    YHIGH: Optional[float]
    YMAX: Optional[float]
    UMIN: Optional[float]
    ULOW: Optional[float]
    UAVG: Optional[float]
    UHIGH: Optional[float]
    UMAX: Optional[float]
    VMIN: Optional[float]
    VLOW: Optional[float]
    VAVG: Optional[float]
    VHIGH: Optional[float]
    VMAX: Optional[float]
    SATMIN: Optional[float]
    SATLOW: Optional[float]
    SATAVG: Optional[float]
    SATHIGH: Optional[float]
    SATMAX: Optional[float]
    HUEMED: Optional[float]
    HUEAVG: Optional[float]
    YDIF: Optional[float]
    UDIF: Optional[float]
    VDIF: Optional[float]
    TOUT: Optional[float]
    VREP: Optional[float]
    BRNG: Optional[float]
    mse_y: Optional[float]
    mse_u: Optional[float]
    mse_v: Optional[float]
    mse_avg: Optional[float]
    psnr_y: Optional[float]
    psnr_u: Optional[float]
    psnr_v: Optional[float]
    psnr_avg: Optional[float]
    Overall_Min_level: Optional[float]
    Overall_Max_level: Optional[float]

@dataclass
class SmpteColorBars:
    YMAX: float
    YMIN: float
    UMIN: float
    UMAX: float
    VMIN: float
    VMAX: float
    SATMIN: float
    SATMAX: float

@dataclass
class QCTParseValues:
    fullTagList: FullTagList
    smpte_color_bars: SmpteColorBars

@dataclass
class SpexConfig:
    filename_values: FilenameValues
    mediainfo_values: Dict[str, Union[MediainfoGeneralValues, MediainfoVideoValues, MediainfoAudioValues]]
    exiftool_values: ExiftoolValues
    ffmpeg_values: Dict[str, Union[FFmpegVideoStream, FFmpegAudioStream, FFmpegFormat]]
    mediatrace_values: MediaTraceValues
    qct_parse_values: QCTParseValues
    signalflow_profiles: Dict[str, Dict] = field(default_factory=dict)

# Output configuration 
@dataclass
class OutputsConfig:
    access_file: bool
    report: bool
    qctools_ext: str

@dataclass
class ChecksumAlgorithm(Enum):
    """Supported checksum algorithms for fixity checking"""
    MD5 = "md5"
    SHA256 = "sha256"

# Fixity configuration 
@dataclass
class FixityConfig:
    check_fixity: bool
    validate_stream_fixity: bool
    embed_stream_fixity: bool
    output_fixity: bool
    overwrite_stream_fixity: bool
    checksum_algorithm: str = "md5"

# Tool-specific configurations 
@dataclass
class BasicToolConfig:
    check_tool: bool
    run_tool: bool

@dataclass
class QCToolsConfig:
    run_tool: bool

@dataclass
class MediaConchConfig:
    mediaconch_policy: str  # This stays as string since it's a filename
    run_mediaconch: bool

@dataclass
class QCTParseToolConfig:
    run_tool: bool
    barsDetection: bool
    evaluateBars: bool
    thumbExport: bool

@dataclass
class ToolsConfig:
    exiftool: BasicToolConfig
    ffprobe: BasicToolConfig
    mediaconch: MediaConchConfig
    mediainfo: BasicToolConfig
    mediatrace: BasicToolConfig
    qctools: QCToolsConfig
    qct_parse: QCTParseToolConfig

@dataclass
class ChecksConfig:
    outputs: OutputsConfig
    fixity: FixityConfig
    tools: ToolsConfig
    validate_filename: bool = True  

@dataclass
class FilenameProfile:
    fn_sections: Dict[str, FilenameSection]
    FileExtension: str

@dataclass
class FilenameConfig:
    filename_profiles: Dict[str, FilenameProfile]

@dataclass
class SignalflowProfile:
    """Profile for signal flow equipment configuration"""
    name: str
    Source_VTR: List[str] = field(default_factory=list)
    TBC_Framesync: List[str] = field(default_factory=list)
    ADC: List[str] = field(default_factory=list)
    Capture_Device: List[str] = field(default_factory=list)
    Computer: List[str] = field(default_factory=list)

@dataclass
class SignalflowConfig:
    """Container for signal flow profiles"""
    signalflow_profiles: Dict[str, SignalflowProfile] = field(default_factory=dict)

@dataclass
class ChecksProfile:
    """Custom profile for checks configuration"""
    name: str
    description: str = ""
    validate_filename: bool = True  
    outputs: OutputsConfig = field(default_factory=lambda: OutputsConfig(
        access_file=False,
        report=False, 
        qctools_ext="qctools.xml.gz"
    ))
    fixity: FixityConfig = field(default_factory=lambda: FixityConfig(
        check_fixity=False,
        validate_stream_fixity=False,
        embed_stream_fixity=False, 
        output_fixity=False,
        overwrite_stream_fixity=False,
        checksum_algorithm="md5" 
    ))
    tools: ToolsConfig = field(default_factory=lambda: ToolsConfig(
        exiftool=BasicToolConfig(check_tool=False, run_tool=False),
        ffprobe=BasicToolConfig(check_tool=False, run_tool=False),
        mediaconch=MediaConchConfig(mediaconch_policy="", run_mediaconch=False),
        mediainfo=BasicToolConfig(check_tool=False, run_tool=False),
        mediatrace=BasicToolConfig(check_tool=False, run_tool=False),
        qctools=QCToolsConfig(run_tool=False),
        qct_parse=QCTParseToolConfig(
            run_tool=False,
            barsDetection=False,
            evaluateBars=False,
            thumbExport=False
        )
    ))

@dataclass
class ChecksProfilesConfig:
    """Container for custom checks profiles"""
    custom_profiles: Dict[str, ChecksProfile] = field(default_factory=dict)

# Add these dataclasses to your config_setup.py file

@dataclass
class ExiftoolProfile:
    """Profile for exiftool expected values configuration"""
    FileType: str
    FileTypeExtension: str
    MIMEType: str
    VideoFrameRate: str
    ImageWidth: str
    ImageHeight: str
    VideoScanType: str
    DisplayWidth: str
    DisplayHeight: str
    DisplayUnit: str
    CodecID: List[str]
    AudioChannels: str
    AudioSampleRate: str
    AudioBitsPerSample: str

@dataclass
class ExiftoolConfig:
    """Container for exiftool profiles"""
    exiftool_profiles: Dict[str, ExiftoolProfile] = field(default_factory=dict)