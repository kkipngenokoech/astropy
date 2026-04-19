import tempfile
import os
from astropy.io import ascii

def test_issue_reproduction():
    # Create a QDP file with lowercase commands
    qdp_content = "read serr 1 2\n1 0.5 1 0.5\n"
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_file = f.name
    
    try:
        # This should work but currently fails with "Unrecognized QDP line: read serr 1 2"
        table = ascii.read(temp_file, format='qdp')
        # If we get here, the bug is fixed
        assert len(table) == 1
        assert len(table.colnames) == 4  # 2 data columns + 2 error columns
    finally:
        # Clean up
        os.unlink(temp_file)