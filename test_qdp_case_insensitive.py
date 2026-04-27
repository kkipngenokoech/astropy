import pytest
import tempfile
import os
from astropy.io import ascii
from astropy.table import Table

def test_issue_reproduction():
    """Test that lowercase QDP commands should be recognized but currently fail."""
    # Create a QDP file content with lowercase commands
    qdp_content = "read serr 1 2\n1 0.5 1 0.5\n"
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_filename = f.name
    
    try:
        # This should work but currently fails with "Unrecognized QDP line: read serr 1 2"
        table = ascii.read(temp_filename, format='qdp')
        
        # If it works correctly, we should get a table with 4 columns:
        # col1, col1_err, col2, col2_err (based on "read serr 1 2" command)
        assert len(table.columns) == 4
        assert 'col1' in table.colnames
        assert 'col1_err' in table.colnames
        assert 'col2' in table.colnames
        assert 'col2_err' in table.colnames
        
        # Check the data values
        assert len(table) == 1
        assert table['col1'][0] == 1
        assert table['col1_err'][0] == 0.5
        assert table['col2'][0] == 1
        assert table['col2_err'][0] == 0.5
        
    finally:
        # Clean up
        os.unlink(temp_filename)