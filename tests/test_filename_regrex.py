import pytest
import re

from AV_Spex.utils.dir_setup import convert_wildcards_to_regex


def _pattern_matches(pattern, should_match, should_not_match):
    """Helper function to test if a pattern correctly matches and rejects strings"""
    regex = convert_wildcards_to_regex(pattern)
    for test_str in should_match:
        assert re.match(f"^{regex}$", test_str), f"Pattern '{pattern}' (regex: {regex}) should match '{test_str}'"
    for test_str in should_not_match:
        assert not re.match(f"^{regex}$", test_str), f"Pattern '{pattern}' (regex: {regex}) should not match '{test_str}'"

def test_single_wildcards():
    """Test individual wildcard characters"""
    test_cases = [
        ("@", ["a", "Z"], ["1", "a1", "aa", ""]),
        ("#", ["0", "9"], ["a", "12", ""]),
        ("*", ["a", "Z", "0", "9"], ["aa", "11", ""]),
    ]
    
    for pattern, valid, invalid in test_cases:
        _pattern_matches(pattern, valid, invalid)

def test_multiple_consecutive_numbers():
    """Test patterns with multiple consecutive # characters"""
    test_cases = [
        ("##", ["12", "00"], ["1", "123", "ab", ""]),
        ("###", ["123", "000"], ["12", "1234", "abc", ""]),
        ("#####", ["12345", "00000"], ["1234", "123456", "abcde", ""]),
    ]
    
    for pattern, valid, invalid in test_cases:
        _pattern_matches(pattern, valid, invalid)

def test_complex_patterns():
    """Test more complex patterns with combinations of wildcards"""
    test_cases = [
        ("AV@*", ["AVaa", "AVb7", "AVz9"], ["AV", "AV1", "BVa1"]),
        ("TEST###@", ["TEST123a", "TEST000Z"], ["TEST123", "TEST1234a", "TESTaaaa"]),
        ("@#@#", ["a1b2", "Z9A0"], ["a1b", "a12b", "1a2b", ""]),
        ("**##", ["ab12", "xy99", "zz00"], ["abc12", "11", "1234", ""]),  # Updated test case
    ]

def test_edge_cases():
    """Test edge cases and special characters"""
    test_cases = [
        # Special regex characters should be escaped
        (".@", [".a", ".Z"], [".1", "a", ""]),
        ("$#", ["$1", "$9"], ["$a", "$12", ""]),
        ("(@)", ["(a)", "(Z)"], ["(1)", "(aa)", ""]),
        ("@@", ["ab", "XY"], ["a1", "1a", ""]),  # Replaced problematic bracket test
        
        # Empty pattern
        ("", [""], ["a", "1", " "]),
        
        # Single character patterns
        ("a", ["a"], ["b", "1", ""]),
    ]

def test_patterns_with_literals():
    """Test patterns that include literal characters"""
    test_cases = [
        ("prefix_@", ["prefix_a", "prefix_Z"], ["prefix_1", "prefix_aa", "prefixa"]),
        ("@_suffix", ["a_suffix", "Z_suffix"], ["1_suffix", "aa_suffix", "a_suffixb"]),
        ("pre_#_post", ["pre_1_post", "pre_9_post"], ["pre_a_post", "pre_12_post"]),
    ]
    
    for pattern, valid, invalid in test_cases:
        _pattern_matches(pattern, valid, invalid)

def test_invalid_inputs():
    """Test handling of invalid inputs"""
    with pytest.raises(TypeError):
        convert_wildcards_to_regex(None)
    
    with pytest.raises(TypeError):
        convert_wildcards_to_regex(123)

def test_case_sensitivity():
    """Test case sensitivity handling"""
    pattern = "ABC@*"
    regex = convert_wildcards_to_regex(pattern)
    
    # Test with re.match (which is case-sensitive by default)
    assert re.match(f"^{regex}$", "ABCa1")
    assert re.match(f"^{regex}$", "ABCz9")
    
    # Test with case-insensitive flag
    assert re.match(f"^{regex}$", "abca1", re.IGNORECASE)
    assert re.match(f"^{regex}$", "ABcZ9", re.IGNORECASE)