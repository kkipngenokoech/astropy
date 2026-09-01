import pytest
from astropy.io.ascii import qdp
from astropy.table import Table
import tempfile
import os

def test_issue_reproduction():
    """Test that QDP parser handles lowercase commands."""
    # Create a QDP file content with lowercase commands
    qdp_content = """! Test QDP file with lowercase commands
read serr 1 2
1.0 0.1 0.05
2.0 0.2 0.10
"""
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_filename = f.name
    
    try:
        # This should work but currently fails due to case sensitivity
        table = Table.read(temp_filename, format='ascii.qdp')
        # If we get here, the bug is fixed
        assert len(table) == 2
        assert len(table.colnames) == 3  # col1, col1_err, col2
    finally:
        os.unlink(temp_filename)