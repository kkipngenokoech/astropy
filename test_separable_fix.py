import numpy as np

# Mock classes
class Model:
    def __init__(self, n_inputs=1, n_outputs=1, separable=True):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.separable = separable
    
    def _calculate_separability_matrix(self):
        return NotImplemented

class CompoundModel(Model):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        
        if op == '&':
            self.n_inputs = left.n_inputs + right.n_inputs
            self.n_outputs = left.n_outputs + right.n_outputs
        
    def _calculate_separability_matrix(self):
        return NotImplemented

class Mapping(Model):
    def __init__(self, mapping):
        self.mapping = mapping
        self.n_inputs = max(mapping) + 1
        self.n_outputs = len(mapping)

# Copy the functions from separable.py
def _compute_n_outputs(left, right):
    if isinstance(left, Model):
        lnout = left.n_outputs
    else:
        lnout = left.shape[0]
    if isinstance(right, Model):
        rnout = right.n_outputs
    else:
        rnout = right.shape[0]
    noutp = lnout + rnout
    return noutp

def _coord_matrix(model, pos, noutp):
    if isinstance(model, Mapping):
        axes = []
        for i in model.mapping:
            axis = np.zeros((model.n_inputs,))
            axis[i] = 1
            axes.append(axis)
        m = np.vstack(axes)
        mat = np.zeros((noutp, model.n_inputs))
        if pos == 'left':
            mat[: model.n_outputs, :model.n_inputs] = m
        else:
            mat[-model.n_outputs:, -model.n_inputs:] = m
        return mat
    if not model.separable:
        mat = np.zeros((noutp, model.n_inputs))
        if pos == 'left':
            mat[:model.n_outputs, : model.n_inputs] = 1
        else:
            mat[-model.n_outputs:, -model.n_inputs:] = 1
    else:
        mat = np.zeros((noutp, model.n_inputs))
        for i in range(model.n_inputs):
            mat[i, i] = 1
        if pos == 'right':
            mat = np.roll(mat, (noutp - model.n_outputs))
    return mat

def _cstack(left, right):
    """Fixed version of _cstack function."""
    noutp = _compute_n_outputs(left, right)

    if isinstance(left, Model):
        cleft = _coord_matrix(left, 'left', noutp)
    else:
        cleft = np.zeros((noutp, left.shape[1]))
        cleft[: left.shape[0], : left.shape[1]] = left
    if isinstance(right, Model):
        cright = _coord_matrix(right, 'right', noutp)
    else:
        cright = np.zeros((noutp, right.shape[1]))
        # FIX: Use 'right' instead of '1'
        cright[-right.shape[0]:, -right.shape[1]:] = right

    return np.hstack([cleft, cright])

def _separable(transform):
    if (transform_matrix := transform._calculate_separability_matrix()) is not NotImplemented:
        return transform_matrix
    elif isinstance(transform, CompoundModel):
        sepleft = _separable(transform.left)
        sepright = _separable(transform.right)
        return _operators[transform.op](sepleft, sepright)
    elif isinstance(transform, Model):
        return _coord_matrix(transform, 'left', transform.n_outputs)

_operators = {'&': _cstack}

# Test the fix
def test_nested_compound_models():
    print("Testing nested compound models separability...")
    
    # Create simple models
    pix2sky = Model(n_inputs=2, n_outputs=2, separable=False)  # Pix2Sky_TAN-like
    linear1 = Model(n_inputs=1, n_outputs=1, separable=True)   # Linear1D-like
    linear2 = Model(n_inputs=1, n_outputs=1, separable=True)   # Linear1D-like
    
    # Create compound model: linear1 & linear2
    cm = CompoundModel('&', linear1, linear2)
    
    # Test simple compound model separability
    cm_sep = _separable(cm)
    print("Simple compound model separability matrix:")
    print(cm_sep)
    
    # Expected: [[1, 0], [0, 1]] (diagonal)
    expected_cm = np.array([[1, 0], [0, 1]])
    assert np.array_equal(cm_sep, expected_cm), f"Expected {expected_cm}, got {cm_sep}"
    print("✓ Simple compound model test passed")
    
    # Create nested compound model: pix2sky & cm
    nested = CompoundModel('&', pix2sky, cm)
    
    # Test nested compound model separability
    nested_sep = _separable(nested)
    print("\nNested compound model separability matrix:")
    print(nested_sep)
    
    # Expected: outputs 0,1 depend on inputs 0,1; outputs 2,3 depend on inputs 2,3 respectively
    expected_nested = np.array([
        [1, 1, 0, 0],  # output 0 depends on inputs 0,1
        [1, 1, 0, 0],  # output 1 depends on inputs 0,1  
        [0, 0, 1, 0],  # output 2 depends only on input 2
        [0, 0, 0, 1]   # output 3 depends only on input 3
    ])
    print("Expected nested separability matrix:")
    print(expected_nested)
    
    assert np.array_equal(nested_sep, expected_nested), f"Expected {expected_nested}, got {nested_sep}"
    print("✓ Nested compound model test passed")
    print("\nAll tests passed! The fix works correctly.")

if __name__ == "__main__":
    test_nested_compound_models()