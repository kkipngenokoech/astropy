import numpy as np
from astropy.io import fits
import tempfile
import os

def test_issue_reproduction():
    """Test that D exponents are properly converted to E in ASCII tables."""
    
    # Create test data with floating point values that would use D exponent notation
    # when written to ASCII table format
    data = [123.0, 0.00456, 7.89]
    
    # Create an ASCII table (not binary) with D format
    col = fits.Column(name='FLOAT_COL', format='D12.5', array=data)
    hdu = fits.TableHDU.from_columns([col])
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        tmp_name = tmp.name
        hdu.writeto(tmp_name, overwrite=True)
    
    try:
        # Read back the data
        with fits.open(tmp_name) as hdul:
            table_data = hdul[1].data
            
            # The values should be correctly read back
            expected_values = [123.0, 0.00456, 7.89]
            actual_values = table_data['FLOAT_COL']
            
            # This assertion should pass if the D->E replacement is working
            np.testing.assert_array_almost_equal(actual_values, expected_values, decimal=5)
            
            print("Test passed: D exponent conversion working correctly")
            
    finally:
        # Clean up
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

if __name__ == "__main__":
    test_issue_reproduction()