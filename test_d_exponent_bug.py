import numpy as np
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.column import ColDefs, Column

def test_issue_reproduction():
    # Create a column with 'D' format (double precision with D exponent)
    # This should trigger the problematic code path
    col = Column(name='test', format='E15.7', array=['1.23D+10', '4.56D-05'])
    coldefs = ColDefs([col])
    
    # Create FITS_rec from columns - this should process the D exponents
    fits_rec = FITS_rec.from_columns(coldefs, character_as_bytes=False)
    
    # The bug is that D exponents are not properly replaced with E exponents
    # because chararray.replace() returns a copy but the result isn't assigned back
    # So the original data should still contain 'D' instead of 'E'
    field_data = fits_rec.field('test')
    
    # If the bug exists, the D exponents won't be replaced properly
    # and we'll still see 'D' in the string representation
    field_str = str(field_data[0])
    assert 'D' not in field_str, f"D exponent not replaced properly: {field_str}"