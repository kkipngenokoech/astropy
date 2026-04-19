import pytest
from astropy.io.ascii.rst import RST

def test_issue_reproduction():
    # This should work but currently fails because RST.__init__() doesn't accept header_rows
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'header_rows'"):
        RST(header_rows=1)