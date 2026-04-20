import numpy as np
from astropy.nddata import NDDataRef

def test_issue_reproduction():
    # Create two NDDataRef objects - one with mask, one without
    data1 = np.array([1, 2, 3])
    mask1 = np.array([True, False, False])
    ndd1 = NDDataRef(data1, mask=mask1)
    
    data2 = np.array([4, 5, 6])
    ndd2 = NDDataRef(data2)  # No mask (mask=None)
    
    # This should fail with TypeError when trying to use bitwise_or on mask and None
    result = ndd1.add(ndd2, handle_mask=np.bitwise_or)