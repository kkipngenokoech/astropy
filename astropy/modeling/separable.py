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
        Indicates the dependence of outputs on inputs. For completely
        independent outputs, the diagonal elements are True and
        off-diagonal elements are False.
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
    Represents the transformation of one coordinate system to another.

    Parameters
    ----------
    model : `Model`
        A simple model.
    pos : str
        Position of this model in the expression tree.
        One of 'left', 'right'.
    n_inputs : int
        Number of inputs.

    Returns
    -------
    array or None
        The array representing the model or None if the model is compound.
    """
    if isinstance(model, Mapping):
        axes = model.mapping
        mat = np.zeros((model.n_outputs, n_inputs))
        for i in range(model.n_outputs):
            mat[i, axes[i]] = 1
        return mat
    elif isinstance(model, CompoundModel):
        return None
    else:
        mat = np.zeros((model.n_outputs, model.n_inputs))
        for i in range(model.n_outputs):
            for j in range(model.n_inputs):
                mat[i, j] = 1
        return mat


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
    n_inputs_left = left.shape[1]
    n_inputs_right = right.shape[1]
    n_outputs_left = left.shape[0]
    n_outputs_right = right.shape[0]

    result = np.zeros((n_outputs_left + n_outputs_right,
                       n_inputs_left + n_inputs_right))
    result[:n_outputs_left, :n_inputs_left] = left
    result[n_outputs_left:, n_inputs_left:] = right
    return result


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
    return np.dot(left, right)


def _cpower(left, right):
    """
    Function corresponding to "**" operation.

    Parameters
    ----------
    left, right : `~astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """
    n_inputs = left.shape[1]
    n_outputs = left.shape[0]
    
    # For A**n, the result is like applying A n times
    # This creates dependencies between all inputs and outputs
    result = np.ones((n_outputs, n_inputs), dtype=bool)
    return result


_operators = {'&': _cstack, '|': _cdot, '**': _cpower}
