import pytest
from astropy.io.ascii.rst import RST

def test_issue_reproduction():
    # This should fail because RST.__init__() doesn't accept header_rows parameter
    # even though its parent FixedWidth class supports it
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'header_rows'"):
        RST(header_rows=2)