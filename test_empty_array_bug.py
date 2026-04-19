import numpy as np
from astropy.wcs import WCS

def test_issue_reproduction():
    # Create a simple WCS object
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1, 1]
    wcs.wcs.cdelt = [1, 1]
    wcs.wcs.crval = [0, 0]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    
    # Test with empty list - this should return empty array but currently fails
    empty_list = []
    result = wcs.all_pix2world(empty_list, 1)
    
    # Should return empty array with proper shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)