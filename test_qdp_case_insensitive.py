import pytest
from astropy.io.ascii.qdp import _line_type

def test_issue_reproduction():
    # Test that lowercase QDP commands should be recognized
    # This should work but currently fails because the regex only matches uppercase
    line_type = _line_type("read serr 1 2")
    assert line_type == "command"