import numpy as np
from astropy.io import fits
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.column import ColDefs, Column
import tempfile
import os

def test_issue_reproduction():
    """
    Test that demonstrates the bug where 'D' exponents in ASCII table data
    are not properly replaced with 'E' exponents due to the replace operation
    not being assigned back to the variable.
    """
    # Create test data with 'D' exponent notation
    data_with_d_exponent = ['1.23D+02', '4.56D-03', '7.89D+00']
    
    # Create a column with ASCII format that should trigger the D->E replacement
    col = Column(name='test_col', format='E10.3', array=data_with_d_exponent)
    coldefs = ColDefs([col])
    
    # Create FITS_rec from columns - this should trigger the problematic code
    fits_rec = FITS_rec.from_columns(coldefs, character_as_bytes=False)
    
    # The bug is that the 'D' exponents should be replaced with 'E' exponents
    # but due to the missing assignment, they remain as 'D'
    
    # Access the field data to trigger any conversion logic
    field_data = fits_rec.field('test_col')
    
    # Create a temporary FITS file to test the full pipeline
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        try:
            # Create an ASCII table HDU with data containing 'D' exponents
            ascii_data = np.array([('1.23D+02',), ('4.56D-03',), ('7.89D+00',)], 
                                dtype=[('col1', 'S10')])
            
            # Create table HDU
            table_hdu = fits.BinTableHDU(data=ascii_data)
            
            # Write to file
            hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
            hdul.writeto(tmp.name, overwrite=True)
            
            # Read back the file
            with fits.open(tmp.name) as hdul_read:
                data_read = hdul_read[1].data
                
                # Check if 'D' exponents are still present (indicating the bug)
                field_values = data_read.field('col1')
                
                # Convert to string for checking
                str_values = [val.decode() if isinstance(val, bytes) else str(val) for val in field_values]
                
                # After the fix, 'D' should be replaced with 'E'
                for val in str_values:
                    assert 'D' not in val, f"Found 'D' exponent in {val}, should be replaced with 'E'"
                    # Should contain 'E' instead
                    if '+' in val or '-' in val:  # Only check if it's scientific notation
                        assert 'E' in val, f"Expected 'E' exponent in {val}"
                        
        finally:
            # Clean up
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    
    print("Test passed: D exponents properly replaced with E exponents")

if __name__ == '__main__':
    test_issue_reproduction()