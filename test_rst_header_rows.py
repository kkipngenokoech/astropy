import pytest
from astropy.io.ascii.rst import RST

def test_issue_reproduction():
    """Test that RST class should accept header_rows parameter but currently fails."""
    # This should work but currently raises TypeError because RST.__init__() 
    # doesn't accept header_rows parameter
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'header_rows'"):
        RST(header_rows=2)