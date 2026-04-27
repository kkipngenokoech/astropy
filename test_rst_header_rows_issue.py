import pytest
from astropy.io.ascii.rst import RST
from astropy.table import QTable
import astropy.units as u
from io import StringIO

def test_issue_reproduction():
    """Test that RST class should accept header_rows parameter."""
    # This should work but currently fails with:
    # TypeError: RST.__init__() got an unexpected keyword argument 'header_rows'
    
    # Try to create RST instance with header_rows parameter
    # This is the core issue - RST.__init__ doesn't accept keyword arguments
    rst_reader = RST(header_rows=['Col1', 'Col2'])
    
    # Create a simple test table to verify the functionality would work
    table_text = '''==== ====
Col1 Col2
==== ====
1    2
3    4
==== ===='''
    
    # This should be able to read the table with the specified header_rows
    result = rst_reader.read(StringIO(table_text))
    
    # Verify the table was read correctly
    assert len(result) == 2
    assert result.colnames == ['Col1', 'Col2']
    assert result['Col1'][0] == 1
    assert result['Col2'][0] == 2