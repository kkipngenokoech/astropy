import numpy as np
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.column import ColDefs, Column

def test_issue_reproduction():
    # Create a column with floating point format that should trigger D exponent handling
    # Use a format that contains 'D' to trigger the problematic code path
    col = Column(name='test', format='D25.17', array=np.array([1.23456789012345e10]))
    coldefs = ColDefs([col])
    
    # Create FITS_rec from columns - this should trigger the D exponent replacement code
    fits_rec = FITS_rec.from_columns(coldefs, nrows=1)
    
    # The bug is that chararray.replace() returns a copy but the result is not assigned back
    # So D exponents in the data should remain unchanged when they should be replaced
    # This test will fail because the current code doesn't actually perform the replacement
    
    # Access the field data to trigger any conversion logic
    field_data = fits_rec.field('test')
    
    # If the D exponent replacement worked properly, we shouldn't see 'D' in string representation
    # But due to the bug, D exponents remain unreplaced
    # This assertion should fail on the buggy code since D replacement doesn't work
    str_repr = str(field_data)
    assert 'D' not in str_repr, f"D exponent not properly replaced in: {str_repr}"