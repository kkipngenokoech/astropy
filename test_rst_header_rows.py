import pytest
from astropy.io.ascii.rst import RST

def test_issue_reproduction():
    """Test that RST class should accept header_rows parameter."""
    # This should not raise a TypeError, but currently does
    rst_writer = RST(header_rows=2)
    assert rst_writer is not None