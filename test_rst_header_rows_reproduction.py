import pytest
from astropy.io.ascii.rst import RST

def test_issue_reproduction():
    """Test that RST class should accept header_rows parameter."""
    # This should work but currently fails with:
    # TypeError: RST.__init__() got an unexpected keyword argument 'header_rows'
    rst_reader = RST(header_rows=['Col1', 'Col2', 'Col3'])
    
    # Verify that the header_rows parameter is properly stored
    assert hasattr(rst_reader, 'header_rows')
    assert rst_reader.header_rows == ['Col1', 'Col2', 'Col3']