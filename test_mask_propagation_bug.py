import numpy as np
from astropy.nddata import NDDataRef

def test_issue_reproduction():
    # Create two NDDataRef objects - one with mask, one without
    data1 = np.array([1, 2, 3])
    mask1 = np.array([True, False, False])
    ndd1 = NDDataRef(data1, mask=mask1)
    
    data2 = np.array([4, 5, 6])
    ndd2 = NDDataRef(data2)  # No mask (mask=None)
    
    # This should fail with TypeError: unsupported operand type(s) for |: 'int' and 'NoneType'
    # when handle_mask=np.bitwise_or tries to operate on mask1 and None
    result = ndd1.add(ndd2, handle_mask=np.bitwise_or)
    
    # Test should pass - the result mask should be the same as mask1
    expected_mask = mask1
    np.testing.assert_array_equal(result.mask, expected_mask)
    
    # Also test the reverse case
    result2 = ndd2.add(ndd1, handle_mask=np.bitwise_or)
    np.testing.assert_array_equal(result2.mask, expected_mask)

if __name__ == "__main__":
    test_issue_reproduction()
    print("Test passed!")