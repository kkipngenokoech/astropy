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
    Create a coordinate matrix for a model.

    Parameters
    ----------
    model : `~astropy.modeling.core.Model`
        A model.
    pos : str
        Position of this model in the expression tree.
        One of 'left', 'right'.
    n_inputs : int
        Number of inputs.

    Returns
    -------
    coord_matrix : ndarray
        A coordinate matrix.
    """
    if isinstance(model, Mapping):
        axes = []
        for i in model.mapping:
            axis = np.zeros((model.n_inputs,))
            axis[i] = 1
            axes.append(axis)
        m = np.vstack(axes)
        mat = np.zeros((model.n_outputs, n_inputs))
        if pos == 'left':
            mat[:, :model.n_inputs] = m
        else:
            mat[:, -model.n_inputs:] = m
        return mat
    elif isinstance(model, CompoundModel):
        # Handle nested compound models recursively
        return _separable(model)
    elif not model.separable:
        # Handle non-separable models
        mat = np.ones((model.n_outputs, n_inputs))
        if pos == 'left':
            mat[:, model.n_inputs:] = 0
        elif pos == 'right':
            mat[:, :-model.n_inputs] = 0
        return mat
    else:
        # Handle separable models
        mat = np.zeros((model.n_outputs, n_inputs))
        if pos == 'left':
            mat[:model.n_outputs, :model.n_inputs] = np.eye(model.n_outputs, model.n_inputs)
        elif pos == 'right':
            mat[:model.n_outputs, -model.n_inputs:] = np.eye(model.n_outputs, model.n_inputs)
        return mat


def _cstack(left, right):
    """
    Function corresponding to '&' operation.

    Parameters
    ----------
    left, right : ndarray or Model
        Coordinate arrays or models.

    Returns
    -------
    result : ndarray
        Result coordinate array.
    """
    if isinstance(left, Model):
        cleft = _coord_matrix(left, 'left', left.n_inputs + right.n_inputs)
    else:
        cleft = np.zeros((left.shape[0], left.shape[1] + right.shape[1]))
        cleft[:left.shape[0], :left.shape[1]] = left

    if isinstance(right, Model):
        cright = _coord_matrix(right, 'right', left.n_inputs + right.n_inputs)
    else:
        cright = np.zeros((right.shape[0], left.shape[1] + right.shape[1]))
        cright[:right.shape[0], left.shape[1]:] = right

    return np.vstack([cleft, cright])


def _cdot(left, right):
    """
    Function corresponding to "|" operation.

    Parameters
    ----------
    left, right : ndarray or Model
        Coordinate arrays or models.

    Returns
    -------
    result : ndarray
        Result coordinate array.
    """
    if isinstance(left, Model):
        cleft = _coord_matrix(left, 'left', right.n_inputs)
    else:
        cleft = left

    if isinstance(right, Model):
        cright = _coord_matrix(right, 'right', right.n_inputs)
    else:
        cright = right

    try:
        return np.dot(cleft, cright)
    except ValueError:
        raise ModelDefinitionError(
            'Models cannot be combined with "|"')


def _cpower(left, right):
    """
    Function corresponding to "**" operation.

    Parameters
    ----------
    left, right : ndarray or Model
        Coordinate arrays or models.

    Returns
    -------
    result : ndarray
        Result coordinate array.
    """
    if isinstance(left, Model):
        cleft = _coord_matrix(left, 'left', left.n_inputs)
    else:
        cleft = left

    return cleft


_operators = {'&': _cstack, '|': _cdot, '**': _cpower}
