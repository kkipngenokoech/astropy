import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

def test_issue_reproduction():
    # Create a simple WCS object
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 100
    header['NAXIS2'] = 100
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = 0.0
    header['CRVAL2'] = 0.0
    header['CRPIX1'] = 50.0
    header['CRPIX2'] = 50.0
    header['CDELT1'] = -0.1
    header['CDELT2'] = 0.1
    
    wcs = WCS(header)
    
    # Test with empty list - this should not fail but return empty array
    empty_list = []
    result = wcs.all_pix2world(empty_list, 1)
    assert len(result) == 0
    
    # Test with empty numpy array - this should not fail but return empty array
    empty_array = np.array([])
    result = wcs.all_pix2world(empty_array, 1)
    assert len(result) == 0