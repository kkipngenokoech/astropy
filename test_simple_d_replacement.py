import numpy as np
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.column import ColDefs, Column, _AsciiColDefs

def test_d_exponent_replacement():
    """
    Simple test to verify that D exponents are replaced with E exponents
    in ASCII table format fields.
    """
    # Create test data with 'D' exponent notation
    test_data = np.array(['1.23D+02', '4.56D-03', '7.89d+00'], dtype='S10')
    
    # Create a column with format that contains 'D'
    col = Column(name='test_col', format='D10.3', array=test_data)
    coldefs = ColDefs([col])
    
    # Make it an ASCII column definition to trigger the D->E replacement
    ascii_coldefs = _AsciiColDefs([col])
    
    # Create FITS_rec
    fits_rec = FITS_rec(test_data.view(np.recarray))
    fits_rec._coldefs = ascii_coldefs
    
    # Access the field - this should trigger the D->E replacement
    field_data = fits_rec.field('test_col')
    
    # Check that D has been replaced with E
    for val in field_data:
        val_str = val.decode() if isinstance(val, bytes) else str(val)
        print(f"Field value: {val_str}")
        assert 'D' not in val_str, f"Found 'D' in {val_str}, should be replaced"
        assert 'd' not in val_str, f"Found 'd' in {val_str}, should be replaced"
        # Should contain E or e
        if '+' in val_str or '-' in val_str:
            assert 'E' in val_str or 'e' in val_str, f"Expected 'E' or 'e' in {val_str}"
    
    print("Test passed: All D/d exponents replaced with E/e")

if __name__ == '__main__':
    test_d_exponent_replacement()