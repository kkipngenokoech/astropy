import tempfile
import os
from astropy.io import ascii
from astropy.table import Table

def test_issue_reproduction():
    # Create a QDP file with lowercase commands
    qdp_content = '''! Test QDP file with lowercase commands
read serr 1 2
! Data follows
1.0 0.1 0.05
2.0 0.2 0.10
3.0 0.3 0.15
'''
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_file = f.name
    
    try:
        # This should work but currently fails due to case-sensitive command matching
        table = Table.read(temp_file, format='ascii.qdp')
        # If we get here, the bug is fixed
        assert len(table) == 3
        assert len(table.colnames) == 3  # col1, col1_err, col2
    finally:
        os.unlink(temp_file)