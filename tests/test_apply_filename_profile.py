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
        
        # Verify get_config was called with the right arguments
        mock_config_mgr.get_config.assert_called_once_with('spex', SpexConfig)
        
        # Verify set_config was called (should be called twice)
        assert mock_config_mgr.set_config.call_count == 2
        
        # Get the config that was passed to set_config
        spex_config = mock_config_mgr.get_config.return_value
        
        # Check that the configuration was updated correctly
        if has_sections:
            assert spex_config.filename_values.fn_sections == sample_profile.fn_sections
        else:
            # If no sections, it should set an empty section
            assert len(spex_config.filename_values.fn_sections) == 1
            assert "section1" in spex_config.filename_values.fn_sections
            assert spex_config.filename_values.fn_sections["section1"].value == ""
            
        if has_extension:
            assert spex_config.filename_values.FileExtension == sample_profile.FileExtension

# Test error handling
def test_apply_filename_profile_error(mock_config_mgr, sample_profile):
    # Make get_config raise an exception
    mock_config_mgr.get_config.side_effect = Exception("Configuration error")
    
    # Apply the patch to replace config_mgr
    with patch('AV_Spex.utils.config_edit.config_mgr', mock_config_mgr):
        # Expect an exception to be raised
        with pytest.raises(Exception) as exc_info:
            apply_filename_profile(sample_profile)
        
        assert "Configuration error" in str(exc_info.value)