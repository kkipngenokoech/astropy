import numpy as np
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.column import ColDefs, Column

def test_issue_reproduction():
    # Create a column with floating point data that has 'D' exponents
    # This should trigger the problematic code path in fitsrec.py
    data_with_d_exponent = np.array(['1.23D+02', '4.56D-03', '7.89D+00'], dtype='S10')
    
    # Create a column definition that would contain 'D' in format
    col = Column(name='test_col', format='E10.3', array=data_with_d_exponent)
    coldefs = ColDefs([col])
    
    # Create FITS_rec from columns - this should process the D exponents
    fits_rec = FITS_rec.from_columns(coldefs, nrows=3)
    
    # The bug is that the replace operation doesn't work in-place
    # So the D exponents should still be present if the bug exists
    # But they should be converted to E exponents if working correctly
    field_data = fits_rec.field('test_col')
    
    # Check if D exponents are still present (indicating the bug)
    # Convert to string for checking since it might be stored as bytes
    field_str = str(field_data)
    
    # If the bug exists, 'D' should still be present in the data
    # This assertion should FAIL on the current buggy code
    assert 'D' not in field_str, "D exponents were not properly replaced with E exponents"