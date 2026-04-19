import pytest
from astropy.io import ascii
from astropy.table import Table
import tempfile
import os

def test_issue_reproduction():
    """Test that QDP files with lowercase commands should be readable."""
    # Create a QDP file content with lowercase commands
    qdp_content = "read serr 1 2\n1 0.5 1 0.5\n"
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_filename = f.name
    
    try:
        # This should work but currently fails with "Unrecognized QDP line: read serr 1 2"
        table = Table.read(temp_filename, format='ascii.qdp')
        
        # If it works, verify the table structure
        assert len(table) == 1
        assert len(table.colnames) == 4  # 2 data columns + 2 error columns
        assert table['col1'][0] == 1
        assert table['col1_err'][0] == 0.5
        assert table['col2'][0] == 1
        assert table['col2_err'][0] == 0.5
        
    finally:
        # Clean up
        os.unlink(temp_filename)