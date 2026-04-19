import pytest
from astropy.table import QTable
import astropy.units as u
from astropy.io import ascii

def test_issue_reproduction():
    """Test that RST format supports header_rows parameter."""
    # Create a simple table
    t = QTable()
    t['col1'] = [1, 2, 3]
    t['col2'] = [4.0, 5.0, 6.0] * u.m
    
    # This should work but currently fails with:
    # TypeError: RST.__init__() got an unexpected keyword argument 'header_rows'
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'header_rows'"):
        ascii.write(t, format='rst', header_rows=['col1', 'col2'])