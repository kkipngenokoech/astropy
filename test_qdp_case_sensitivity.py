import pytest
from astropy.io import ascii
from astropy.io.ascii.qdp import _line_type

def test_issue_reproduction():
    # Test that lowercase QDP commands should be recognized but currently fail
    # This reproduces the exact issue described: "read serr 1 2" should work
    
    # Test the _line_type function directly with lowercase command
    with pytest.raises(ValueError, match="Unrecognized QDP line"):
        _line_type("read serr 1 2")
    
    # Test reading a QDP file with lowercase commands
    qdp_content = "read serr 1 2\n1 0.5 1 0.5\n"
    
    with pytest.raises(ValueError, match="Unrecognized QDP line"):
        ascii.read(qdp_content, format='qdp')