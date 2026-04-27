#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix, _separable
from astropy.modeling.core import CompoundModel

# Create the models as described in the issue
cm = m.Linear1D(10) & m.Linear1D(5)
print("Simple compound model (cm):")
print(f"  Type: {type(cm)}")
print(f"  n_inputs: {cm.n_inputs}, n_outputs: {cm.n_outputs}")
print(f"  Separability matrix:\n{separability_matrix(cm)}")
print()

# Create the nested compound model
nested = m.Pix2Sky_TAN() & cm
print("Nested compound model (nested):")
print(f"  Type: {type(nested)}")
print(f"  n_inputs: {nested.n_inputs}, n_outputs: {nested.n_outputs}")
print(f"  Left: {type(nested.left)} (n_inputs={nested.left.n_inputs}, n_outputs={nested.left.n_outputs})")
print(f"  Right: {type(nested.right)} (n_inputs={nested.right.n_inputs}, n_outputs={nested.right.n_outputs})")
print(f"  Separability matrix:\n{separability_matrix(nested)}")
print()

# Let's also test the internal _separable function
print("Internal _separable results:")
print(f"  cm: \n{_separable(cm)}")
print(f"  nested: \n{_separable(nested)}")