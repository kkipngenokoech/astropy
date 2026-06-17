import pytest
from astropy.io.ascii.rst import RST

def test_issue_reproduction():
    """Test that RST class should accept header_rows parameter."""
    # This should work but currently fails with TypeError
    rst_reader = RST(header_rows=['Col1', 'Col2', 'Col3'])
    assert rst_reader is not None