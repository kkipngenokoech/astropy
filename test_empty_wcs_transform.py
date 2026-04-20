import numpy as np
from astropy.wcs import WCS

def test_issue_reproduction():
    # Create a simple WCS object
    wcs = WCS(naxis=2)
    
    # Test with empty list - this should return empty array but currently fails
    result = wcs.all_pix2world([], 1)
    assert len(result) == 0
    
    # Test with empty numpy array - this should also return empty array but currently fails
    empty_array = np.array([])
    result = wcs.all_pix2world(empty_array, 1)
    assert len(result) == 0