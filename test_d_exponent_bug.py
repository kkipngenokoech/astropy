import numpy as np
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.column import ColDefs, Column

def test_issue_reproduction():
    # Create a column with 'D' exponent format that should be processed
    # This tests the bug where chararray.replace() returns a copy but the code
    # doesn't assign it back, so 'D' exponents are not actually replaced
    
    # Create test data with 'D' exponent notation
    test_data = np.array(['1.23D+02', '4.56D-01', '7.89D+00'], dtype='S10')
    
    # Create a column definition for floating point data
    col = Column(name='test_col', format='E10.3', array=test_data)
    coldefs = ColDefs([col])
    
    # Create FITS_rec from columns - this should trigger the D exponent processing
    fits_rec = FITS_rec.from_columns(coldefs, character_as_bytes=True)
    
    # The bug is that 'D' exponents are not being replaced with 'E' exponents
    # because chararray.replace() returns a copy that's not assigned back
    # So the original data should still contain 'D' when it should contain 'E'
    
    # Get the raw field data
    field_data = fits_rec.field('test_col')
    
    # Convert to string to check content
    if hasattr(field_data, 'tobytes'):
        field_str = field_data.tobytes().decode('ascii', errors='ignore')
    else:
        field_str = str(field_data)
    
    # The bug means 'D' should still be present when it should have been replaced with 'E'
    # This assertion will fail on the current buggy code because 'D' is not being replaced
    assert 'D' not in field_str, f"'D' exponents were not replaced with 'E': {field_str}"
    assert 'E' in field_str, f"Expected 'E' exponents but found: {field_str}"