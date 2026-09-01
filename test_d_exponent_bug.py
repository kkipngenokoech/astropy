import numpy as np
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.column import ColDefs, Column

def test_issue_reproduction():
    # Create a column with floating point data that contains 'D' exponents
    # This simulates data that would come from a FITS file with FORTRAN-style exponents
    data_with_d_exponent = np.array(['1.23D+02', '4.56D-03', '7.89D+00'], dtype='S10')
    
    # Create a column definition that would trigger the 'D' exponent replacement code
    col = Column(name='test_col', format='E10.3', array=data_with_d_exponent)
    coldefs = ColDefs([col])
    
    # Create a FITS_rec from the columns
    fits_rec = FITS_rec.from_columns(coldefs, character_as_bytes=True)
    
    # Access the field data - this should trigger any conversion logic
    field_data = fits_rec.field('test_col')
    
    # If the bug exists, the 'D' exponents won't be properly converted to 'E'
    # The data should be converted to proper float values, but if the replace
    # operation isn't assigned back, the 'D' characters will remain
    field_str = str(field_data)
    
    # This assertion will fail if the 'D' exponents are not properly converted
    # because the replace() call doesn't assign the result back to the original array
    assert 'D' not in field_str, f"'D' exponents were not converted in field data: {field_str}"