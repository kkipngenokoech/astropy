import pytest
from astropy.io.ascii import qdp
from astropy.table import Table
import tempfile
import os

def test_issue_reproduction():
    """Test that QDP format handles lowercase commands like 'read serr'."""
    
    # Create a QDP file content with lowercase command
    qdp_content = "read serr 1 2\n1 0.5 1 0.5\n"
    
    # Test the _line_type function directly with lowercase command
    lowercase_command_line = "read serr 1 2"
    
    # This should not raise an exception but currently does
    with pytest.raises(ValueError, match="Unrecognized QDP line: read serr 1 2"):
        qdp._line_type(lowercase_command_line)
    
    # Test reading a complete QDP file with lowercase commands
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_filename = f.name
    
    try:
        # This should work but currently raises ValueError
        with pytest.raises(ValueError, match="Unrecognized QDP line: read serr 1 2"):
            table = Table.read(temp_filename, format='ascii.qdp')
    finally:
        os.unlink(temp_filename)
    
    # Test that uppercase version works (this should pass)
    uppercase_command_line = "READ SERR 1 2"
    result = qdp._line_type(uppercase_command_line)
    assert result == "command"
    
    # Test reading uppercase QDP file works
    uppercase_qdp_content = "READ SERR 1 2\n1 0.5 1 0.5\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(uppercase_qdp_content)
        temp_filename = f.name
    
    try:
        # This should work and does work
        table = Table.read(temp_filename, format='ascii.qdp')
        assert len(table) == 1
        assert len(table.columns) == 4  # 2 data columns + 2 error columns
        assert table.colnames == ['col1', 'col1_err', 'col2', 'col2_err']
    finally:
        os.unlink(temp_filename)