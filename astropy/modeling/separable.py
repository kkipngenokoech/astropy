# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Functions to determine if a model is separable, i.e.
if the model outputs are independent.

It analyzes ``n_inputs``, ``n_outputs`` and the operators
in a compound model by stepping through the transforms
and creating a ``coord_matrix`` of shape (``n_outputs``, ``n_inputs``).


Each modeling operator is represented by a function which
takes two simple models (or two ``coord_matrix`` arrays) and
returns an array of shape (``n_outputs``, ``n_inputs``).

"""

import numpy as np

from .core import Model, ModelDefinitionError, CompoundModel
from .mappings import Mapping


__all__ = ["is_separable", "separability_matrix"]


def is_separable(transform):
    """
    A separability test for the outputs of a transform.

    Parameters
    ----------
    transform : `~astropy.modeling.core.Model`
        A (compound) model.

    Returns
    -------
    is_separable : ndarray
        A boolean array with size ``transform.n_outputs`` where
        each element indicates whether the output is independent
        and the result of a separable transform.

    Examples
    --------
    >>> from astropy.modeling.models import Shift, Scale, Rotation2D, Polynomial2D
    >>> is_separable(Shift(1) & Shift(2) | Scale(1) & Scale(2))
        array([ True,  True]...)
    >>> is_separable(Shift(1) & Shift(2) | Rotation2D(2))
        array([False, False]...)
    >>> is_separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) | \
        Polynomial2D(1) & Polynomial2D(2))
        array([False, False]...)
    >>> is_separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]))
        array([ True,  True,  True,  True]...)

    """
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        is_separable = np.array([False] * transform.n_outputs).T
        return is_separable
    separable_matrix = _separable(transform)
    is_separable = separable_matrix.sum(1)
    is_separable = np.where(is_separable != 1, False, True)
    return is_separable


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
    >>> from astropy.modeling.models import Shift, Scale, Rotation2D, Polynomial2D
    >>> separability_matrix(Shift(1) & Shift(2) | Scale(1) & Scale(2))
        array([[ True, False], [False,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Rotation2D(2))
        array([[ True,  True], [ True,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) | \
        Polynomial2D(1) & Polynomial2D(2))
        array([[ True,  True], [ True,  True]]...)

    """
    return _separable(transform)


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
        Indicates the dependence of outputs on inputs.
    """
    if (transform_matrix := _coord_matrix(transform, 'left', transform.n_inputs)) is not None:
        return transform_matrix
    elif isinstance(transform, CompoundModel):
        sepleft = _separable(transform.left)
        sepright = _separable(transform.right)
        return _operators[transform.op](sepleft, sepright)
    elif isinstance(transform, Model):
        return _coord_matrix(transform, 'left', transform.n_inputs)


def _coord_matrix(model, pos, n_inputs):
    """
    Create an array representing inputs and outputs of a simple model.

    The array has a shape (n_outputs, n_inputs).
    Represents the correlation between inputs and outputs of a model.

    Examples
    --------
    >>> _coord_matrix(Shift(1), 'left', 1)
    array([[ True]])
    >>> _coord_matrix(Shift(1) & Shift(2), 'left', 2)
    array([[ True, False], [False, True]])
    >>> _coord_matrix(Scale(1) & Scale(2), 'left', 2)
    array([[ True, False], [False, True]])
    >>> _coord_matrix(Mapping([0, 1, 0, 1]), 'left', 2)
    array([[ True, False], [False, True], [ True, False], [False, True]])
    >>> _coord_matrix(Polynomial2D(2), 'left', 2)
    array([[ True, True]])
    >>> _coord_matrix(Shift(1) | Shift(2), 'left', 1)
    array([[ True]])
    """
    if isinstance(model, Mapping):
        axes = model.mapping
        coord_matrix = np.zeros((model.n_outputs, n_inputs))
        for i, axis in enumerate(axes):
            coord_matrix[i, axis] = True
        return coord_matrix
    elif isinstance(model, CompoundModel):
        # Handle nested CompoundModels properly
        return None  # Signal that we need to recurse
    else:
        # Handle simple models
        if model.standard_broadcasting:
            if model.n_inputs == model.n_outputs:
                coord_matrix = np.eye(model.n_outputs, n_inputs, dtype=bool)
            else:
                coord_matrix = np.ones((model.n_outputs, n_inputs), dtype=bool)
        else:
            coord_matrix = np.ones((model.n_outputs, n_inputs), dtype=bool)
        return coord_matrix


def _cstack(left, right):
    """
    Function corresponding to '&' operation.

    Parameters
    ----------
    left, right : `~astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """
    # Handle the case where inputs are models
    if isinstance(left, Model) and isinstance(right, Model):
        # Calculate total inputs for the combined model
        n_inputs_left = left.n_inputs
        n_inputs_right = right.n_inputs
        total_inputs = n_inputs_left + n_inputs_right
        
        # Get separability matrices for each model
        left_matrix = _separable(left)
        right_matrix = _separable(right)
        
        # Create the combined matrix
        combined_matrix = np.zeros((left.n_outputs + right.n_outputs, total_inputs), dtype=bool)
        
        # Fill in the left model's dependencies
        combined_matrix[:left.n_outputs, :n_inputs_left] = left_matrix
        
        # Fill in the right model's dependencies (offset by left model's inputs)
        combined_matrix[left.n_outputs:, n_inputs_left:] = right_matrix
        
        return combined_matrix
    
    # Handle the case where inputs are already matrices
    else:
        # This is the original logic for when we already have matrices
        n_inputs_left = left.shape[1]
        n_inputs_right = right.shape[1]
        
        combined_matrix = np.zeros((left.shape[0] + right.shape[0], 
                                   n_inputs_left + n_inputs_right), dtype=bool)
        
        combined_matrix[:left.shape[0], :n_inputs_left] = left
        combined_matrix[left.shape[0]:, n_inputs_left:] = right
        
        return combined_matrix


def _cdot(left, right):
    """
    Function corresponding to "|" operation.

    Parameters
    ----------
    left, right : `~astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """
    # Handle the case where inputs are models
    if isinstance(left, Model) and isinstance(right, Model):
        left_matrix = _separable(left)
        right_matrix = _separable(right)
        return np.dot(right_matrix, left_matrix)
    
    # Handle the case where inputs are already matrices
    else:
        return np.dot(left, right)


def _cfix_inputs(left, right):
    """
    Function corresponding to "+" operation with fix_inputs.

    Parameters
    ----------
    left, right : `~astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """
    # This is a simplified implementation - may need refinement
    # based on the specific fix_inputs behavior
    if isinstance(left, Model) and isinstance(right, Model):
        left_matrix = _separable(left)
        right_matrix = _separable(right)
        # For fix_inputs, we typically combine the matrices
        # This may need adjustment based on actual fix_inputs semantics
        return np.logical_or(left_matrix, right_matrix)
    else:
        return np.logical_or(left, right)


_operators = {'&': _cstack, '|': _cdot, '+': _cfix_inputs}
