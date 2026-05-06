import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict

# Import the necessary classes and the function under test
from AV_Spex.utils.config_edit import apply_filename_profile
from AV_Spex.utils.config_setup import FilenameProfile, FilenameSection, SpexConfig, FilenameValues
from AV_Spex.utils.config_manager import ConfigManager

# Create mock classes for testing
@pytest.fixture
def mock_config_mgr():
    mock = MagicMock()

    # Setup the get_config method to return a properly structured SpexConfig
    spex_config = MagicMock(spec=SpexConfig)
    filename_values = MagicMock(spec=FilenameValues)
    filename_values.fn_sections = {}
    filename_values.FileExtension = ".txt"
    spex_config.filename_values = filename_values

    mock.get_config.return_value = spex_config
    return mock

@pytest.fixture
def sample_profile():
    # Create a sample FilenameProfile for testing
    sections = {
        "section1": FilenameSection(value="test", section_type="literal"),
        "section2": FilenameSection(value="date", section_type="date")
    }
    return FilenameProfile(fn_sections=sections, FileExtension=".csv")

# Patch the config_mgr in the module under test
@pytest.mark.parametrize("has_sections,has_extension", [
    (True, True),    # Both sections and extension
    (True, False),   # Only sections
    (False, True),   # Only extension
    (False, False),  # Neither
])
def test_apply_filename_profile(mock_config_mgr, sample_profile, has_sections, has_extension):
    # Setup the test case
    if not has_sections:
        sample_profile.fn_sections = {}

    if not has_extension:
        sample_profile.FileExtension = ""

    # Apply the patch to replace config_mgr
    with patch('AV_Spex.utils.config_edit.config_mgr', mock_config_mgr):
        # Call the function under test
        apply_filename_profile(sample_profile)

        # apply_filename_profile now uses replace_config_section twice:
        # once for fn_sections and once for FileExtension.
        assert mock_config_mgr.replace_config_section.call_count == 2

        calls = mock_config_mgr.replace_config_section.call_args_list
        call_targets = {c.args[1]: c.args[2] for c in calls}

        assert 'filename_values.fn_sections' in call_targets
        assert 'filename_values.FileExtension' in call_targets

        sections_written = call_targets['filename_values.fn_sections']
        extension_written = call_targets['filename_values.FileExtension']

        if has_sections:
            # Each section should be serialized as a dict with value/section_type
            assert set(sections_written.keys()) == set(sample_profile.fn_sections.keys())
            for key, section in sample_profile.fn_sections.items():
                assert sections_written[key]['value'] == section.value
                assert sections_written[key]['section_type'] == section.section_type
        else:
            assert sections_written == {}

        assert extension_written == sample_profile.FileExtension

        # Function should also refresh and re-read the config at the end
        mock_config_mgr.refresh_configs.assert_called_once()
        mock_config_mgr.get_config.assert_called_with('spex', SpexConfig)

# Test error handling
def test_apply_filename_profile_error(mock_config_mgr, sample_profile):
    # Make replace_config_section raise an exception to simulate a failure
    mock_config_mgr.replace_config_section.side_effect = Exception("Configuration error")

    # Apply the patch to replace config_mgr
    with patch('AV_Spex.utils.config_edit.config_mgr', mock_config_mgr):
        # Expect an exception to be raised
        with pytest.raises(Exception) as exc_info:
            apply_filename_profile(sample_profile)

        assert "Configuration error" in str(exc_info.value)