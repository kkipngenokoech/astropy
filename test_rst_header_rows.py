import pytest
from astropy.io.ascii.rst import RST

def test_issue_reproduction():
    # This should work but currently fails with TypeError
    rst_writer = RST(header_rows=2)
    assert rst_writer is not None