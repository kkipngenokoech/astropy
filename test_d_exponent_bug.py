import numpy as np
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.column import ColDefs, Column

def test_issue_reproduction():
    # Create a column with 'D' format (double precision with D exponent)
    # This should trigger the buggy code path in _convert_ascii_table
    col = Column(name='test', format='E15.7', array=['1.23D+05', '4.56D-03'])
    coldefs = ColDefs([col])
    
    # Create FITS_rec from columns - this should trigger ASCII table conversion
    fits_rec = FITS_rec.from_columns(coldefs, character_as_bytes=False)
    
    # The bug is that 'D' exponents are not properly replaced with 'E'
    # because chararray.replace() result is not assigned back
    # So the data should still contain 'D' instead of being converted to 'E'
    field_data = fits_rec.field('test')
    
    # If the bug exists, the 'D' characters should still be present
    # because the replace operation result wasn't assigned back
    # This assertion will fail when the bug is fixed
    assert b'D' in field_data.tobytes(), "Bug reproduction failed - 'D' exponents should still be present due to the replace() bug"