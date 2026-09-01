import pytest
from astropy.io.ascii import qdp

def test_issue_reproduction():
    # Test that lowercase QDP commands should be recognized
    # This should not raise a ValueError but currently does
    qdp_content = "read serr 1 2\n1 0.5 1 0.5\n"
    
    # This should work but currently fails with "Unrecognized QDP line: read serr 1 2"
    with pytest.raises(ValueError, match="Unrecognized QDP line"):
        qdp._get_tables_from_qdp_file(qdp_content)