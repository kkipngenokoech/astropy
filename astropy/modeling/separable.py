"""
Functions for determining model separability.
"""

import numpy as np
from .core import Model, CompoundModel


def separability_matrix(transform):
    """
    Compute the correlation between outputs and inputs.

    Parameters
    ----------
    transform : `~astropy.modeling.core.Model`
        A (compound) model.

    Returns
    -------
    separable_matrix : ndarray
        A boolean correlation matrix of shape (n_outputs, n_inputs).
        Indicates the dependence of outputs on inputs. For completely
        independent outputs, the diagonal elements are True and
        off-diagonal elements are False.

    Examples
    --------
    >>> from astropy.modeling.models import Shift, Scale, Rotation2D
    >>> separability_matrix(Shift(1) & Shift(2) | Scale(1) & Scale(2))
    array([[ True, False],
           [False,  True]])
    >>> separability_matrix(Shift(1) & Shift(2) | Rotation2D(2))
    array([[ True,  True],
           [ True,  True]])
    """
    if transform.n_inputs == 1 and transform.n_outputs == 1:
        return np.array([[True]])

    separable_matrix = _separable(transform)
    separable_matrix = np.asarray(separable_matrix, dtype=bool)
    return separable_matrix


def _separable(transform):
    """
    Calculate the separability of outputs.

    Parameters
    ----------
    transform : `~astropy.modeling.core.Model`
        A transform (usually a compound model).

    Returns
    -------
    separable_matrix : ndarray
        A boolean correlation matrix of shape (n_outputs, n_inputs).
    """
    if (transform_matrix := _compute_n_outputs(transform, 'left')) is not None:
        return transform_matrix
    elif (transform_matrix := _compute_n_outputs(transform, 'right')) is not None:
        return transform_matrix
    else:
        return _coord_matrix(transform, 'left', None)


def _compute_n_outputs(model, pos):
    """
    Calculate the number of outputs of a model.
    If not possible returns None.

    Parameters
    ----------
    model : `~astropy.modeling.core.Model`
    pos : str
        Position of this model in the expression tree.
        One of ['left', 'right'].
    """
    if isinstance(model, CompoundModel):
        if model.op == '&':
            left = _compute_n_outputs(model.left, 'left')
            right = _compute_n_outputs(model.right, 'right')
            if left is None or right is None:
                return None
            else:
                # For parallel composition (&), concatenate the matrices
                # Fix: Handle nested compound models properly by flattening
                if isinstance(left, (list, tuple)):
                    left = np.array(left)
                if isinstance(right, (list, tuple)):
                    right = np.array(right)
                
                # Ensure we have 2D arrays
                if left.ndim == 1:
                    left = left.reshape(1, -1)
                if right.ndim == 1:
                    right = right.reshape(1, -1)
                    
                # Create block diagonal matrix
                n_left_out, n_left_in = left.shape
                n_right_out, n_right_in = right.shape
                
                result = np.zeros((n_left_out + n_right_out, n_left_in + n_right_in), dtype=bool)
                result[:n_left_out, :n_left_in] = left
                result[n_left_out:, n_left_in:] = right
                
                return result
        elif model.op == '|':
            left = _compute_n_outputs(model.left, 'left')
            right = _compute_n_outputs(model.right, 'right')
            if left is None or right is None:
                return None
            else:
                # For serial composition (|), matrix multiply
                if isinstance(left, (list, tuple)):
                    left = np.array(left)
                if isinstance(right, (list, tuple)):
                    right = np.array(right)
                    
                # Ensure we have 2D arrays
                if left.ndim == 1:
                    left = left.reshape(1, -1)
                if right.ndim == 1:
                    right = right.reshape(1, -1)
                    
                return np.dot(right, left)
        elif model.op == '+':
            # Addition - outputs depend on same inputs
            left = _compute_n_outputs(model.left, 'left')
            right = _compute_n_outputs(model.right, 'right')
            if left is None or right is None:
                return None
            else:
                if isinstance(left, (list, tuple)):
                    left = np.array(left)
                if isinstance(right, (list, tuple)):
                    right = np.array(right)
                    
                # For addition, take logical OR of dependencies
                return np.logical_or(left, right)
        elif model.op == '*':
            # Multiplication - outputs depend on same inputs  
            left = _compute_n_outputs(model.left, 'left')
            right = _compute_n_outputs(model.right, 'right')
            if left is None or right is None:
                return None
            else:
                if isinstance(left, (list, tuple)):
                    left = np.array(left)
                if isinstance(right, (list, tuple)):
                    right = np.array(right)
                    
                # For multiplication, take logical OR of dependencies
                return np.logical_or(left, right)
        elif model.op == '**':
            # Power - same as multiplication
            left = _compute_n_outputs(model.left, 'left')
            right = _compute_n_outputs(model.right, 'right')
            if left is None or right is None:
                return None
            else:
                if isinstance(left, (list, tuple)):
                    left = np.array(left)
                if isinstance(right, (list, tuple)):
                    right = np.array(right)
                    
                return np.logical_or(left, right)
    else:
        # Simple model - create identity-like matrix
        return np.ones((model.n_outputs, model.n_inputs), dtype=bool)


def _coord_matrix(model, pos, coord):
    """
    Create a coordinate matrix for a model.
    
    Parameters
    ----------
    model : `~astropy.modeling.core.Model`
    pos : str
        Position of this model in the expression tree.
    coord : int or None
        Coordinate index.
        
    Returns
    -------
    coord_matrix : ndarray
        Boolean matrix indicating input-output dependencies.
    """
    if isinstance(model, CompoundModel):
        if model.op == '&':
            left = _coord_matrix(model.left, 'left', coord)
            right = _coord_matrix(model.right, 'right', coord)
            
            # Ensure arrays
            if isinstance(left, (list, tuple)):
                left = np.array(left)
            if isinstance(right, (list, tuple)):
                right = np.array(right)
                
            # Ensure 2D
            if left.ndim == 1:
                left = left.reshape(1, -1)
            if right.ndim == 1:
                right = right.reshape(1, -1)
                
            # Create block diagonal
            n_left_out, n_left_in = left.shape
            n_right_out, n_right_in = right.shape
            
            result = np.zeros((n_left_out + n_right_out, n_left_in + n_right_in), dtype=bool)
            result[:n_left_out, :n_left_in] = left
            result[n_left_out:, n_left_in:] = right
            
            return result
        elif model.op == '|':
            left = _coord_matrix(model.left, 'left', coord)
            right = _coord_matrix(model.right, 'right', coord)
            
            if isinstance(left, (list, tuple)):
                left = np.array(left)
            if isinstance(right, (list, tuple)):
                right = np.array(right)
                
            if left.ndim == 1:
                left = left.reshape(1, -1)
            if right.ndim == 1:
                right = right.reshape(1, -1)
                
            return np.dot(right, left)
        else:
            # Other operations
            left = _coord_matrix(model.left, 'left', coord)
            right = _coord_matrix(model.right, 'right', coord)
            
            if isinstance(left, (list, tuple)):
                left = np.array(left)
            if isinstance(right, (list, tuple)):
                right = np.array(right)
                
            return np.logical_or(left, right)
    else:
        # Simple model
        return np.ones((model.n_outputs, model.n_inputs), dtype=bool)
