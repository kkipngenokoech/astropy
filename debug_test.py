import sys
sys.path.insert(0, '.')

import numpy as np
from astropy.modeling.separable import _separable, _cstack
from astropy.modeling.models import Linear1D, Shift

# Test the _cstack function directly
left_model = Shift(1)  # Simple model
right_array = np.array([[True, False], [False, True]])  # Separability matrix from compound model

print("Left model (Shift):")
left_sep = _separable(left_model)
print(left_sep)

print("\nRight array (compound model separability):")
print(right_array)

print("\n_cstack result:")
result = _cstack(left_sep, right_array)
print(result)

print("\nExpected result should preserve the right array structure")