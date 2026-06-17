import pytest
from astropy.io.ascii import qdp

def test_issue_reproduction():
    """Test that lowercase QDP commands should be recognized but currently fail."""
    # Create QDP content with lowercase command
    qdp_content = "read serr 1 2\n1 0.5 1 0.5\n"
    
    # This should work but currently raises ValueError due to case sensitivity
    with pytest.raises(ValueError, match="Unrecognized QDP line: read serr 1 2"):
        qdp._get_tables_from_qdp_file(qdp_content)