import pytest
from astropy.io.ascii.rst import RST

def test_issue_reproduction():
    # This should work but currently fails because RST.__init__() doesn't accept header_rows
    rst_reader = RST(header_rows=['Col1', 'Col2', 'Col3'])
    assert rst_reader is not None