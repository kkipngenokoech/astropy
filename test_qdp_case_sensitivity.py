import pytest
from astropy.io import ascii
from astropy.table import Table
import tempfile
import os

def test_issue_reproduction():
    """Test that QDP files with lowercase commands should be readable."""
    # Create a QDP file content with lowercase commands
    qdp_content = """! Test QDP file with lowercase commands
read serr 1 2
! Column names: x x_err y y_err
1.0 0.1 2.0 0.2
2.0 0.15 3.0 0.25
"""
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_filename = f.name
    
    try:
        # This should work but currently fails due to case sensitivity
        table = Table.read(temp_filename, format='ascii.qdp')
        
        # Verify the table was read correctly
        assert len(table) == 2
        assert len(table.colnames) == 4  # x, x_err, y, y_err
        assert 'col1_err' in table.colnames
        assert 'col2_err' in table.colnames
        
    finally:
        # Clean up
        os.unlink(temp_filename)