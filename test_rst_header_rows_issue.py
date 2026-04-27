import pytest
from astropy.table import QTable
import astropy.units as u
from astropy.io import ascii
from io import StringIO

def test_issue_reproduction():
    """Test that RST format supports header_rows parameter."""
    # Create a simple table
    t = QTable()
    t['col1'] = [1, 2, 3]
    t['col2'] = [4.0, 5.0, 6.0] * u.m
    t['col3'] = ['a', 'b', 'c']
    
    # This should work but currently fails with:
    # TypeError: RST.__init__() got an unexpected keyword argument 'header_rows'
    output = StringIO()
    
    # The issue is that RST class doesn't accept header_rows parameter
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'header_rows'"):
        ascii.write(t, output, format='rst', header_rows=['units'])
    
    # Also test direct instantiation which should also fail
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'header_rows'"):
        rst_writer = ascii.RST(header_rows=['units'])