import pytest
from astropy.io.ascii import RST
from astropy.table import QTable
import astropy.units as u
from io import StringIO

def test_issue_reproduction():
    """Test that RST class should accept header_rows parameter."""
    # This should work but currently fails because RST.__init__() doesn't accept header_rows
    with pytest.raises(TypeError, match="RST.__init__\(\) got an unexpected keyword argument 'header_rows'"):
        rst_reader = RST(header_rows=['Col1', 'Col2', 'Col3'])
    
    # Also test that we can't pass header_rows when reading a table
    table_content = """==== ===== ======
Col1  Col2  Col3
==== ===== ======
  1    2.3  Hello
  2    4.5  Worlds
==== ===== ======"""
    
    # This should also fail with the same error when trying to use header_rows
    with pytest.raises(TypeError, match="RST.__init__\(\) got an unexpected keyword argument 'header_rows'"):
        from astropy.io import ascii
        ascii.read(StringIO(table_content), format='rst', header_rows=['CustomCol1', 'CustomCol2', 'CustomCol3'])