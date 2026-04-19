import numpy as np
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.column import ColDefs, Column

def test_issue_reproduction():
    # Create a column with floating point data containing 'D' exponents
    # This simulates ASCII table data that would contain 'D' instead of 'E' for exponents
    data_with_d_exponent = np.array(['1.23D+02', '4.56D-01', '7.89D+00'], dtype='S8')
    
    # Create a column definition for floating point data
    col = Column(name='test_col', format='E8.2', array=data_with_d_exponent)
    coldefs = ColDefs([col])
    
    # Create FITS_rec from the column - this should trigger the D exponent handling
    fits_rec = FITS_rec.from_columns(coldefs, character_as_bytes=True)
    
    # Access the field to trigger any conversion logic
    field_data = fits_rec.field('test_col')
    
    # The bug is that 'D' exponents are not properly replaced with 'E' exponents
    # So the conversion to float should fail or produce incorrect results
    # If the bug exists, the original 'D' characters should still be present
    raw_field = fits_rec._get_raw_field('test_col')
    
    # Check if 'D' characters are still present (indicating the bug)
    # The replace operation should have converted 'D' to 'E' but didn't due to the bug
    assert b'D' not in raw_field.tobytes(), "D exponents were not properly replaced with E exponents"