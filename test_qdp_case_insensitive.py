import pytest
from astropy.io.ascii.qdp import _line_type

def test_issue_reproduction():
    # Test that lowercase QDP commands should be recognized
    # This should work but currently fails because _line_type only accepts uppercase commands
    line = "read serr 1 2"
    result = _line_type(line)
    assert result == "command"