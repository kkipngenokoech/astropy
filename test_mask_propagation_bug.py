import numpy as np
from astropy.nddata import NDDataRef

def test_issue_reproduction():
    """Test that mask propagation fails when one operand has no mask and handle_mask=np.bitwise_or."""
    # Create two NDDataRef objects - one with mask, one without
    data1 = np.array([1, 2, 3])
    mask1 = np.array([True, False, False])
    ndd1 = NDDataRef(data1, mask=mask1)
    
    data2 = np.array([4, 5, 6])
    ndd2 = NDDataRef(data2)  # No mask (mask=None)
    
    # This should work but currently fails with TypeError
    # The expected behavior is that the result should have mask1 as the mask
    result = ndd1.add(ndd2, handle_mask=np.bitwise_or)
    
    # The result should have the mask from ndd1 since ndd2 has no mask
    expected_mask = mask1
    np.testing.assert_array_equal(result.mask, expected_mask)
    
    # Also test the reverse operation
    result2 = ndd2.add(ndd1, handle_mask=np.bitwise_or)
    np.testing.assert_array_equal(result2.mask, expected_mask)