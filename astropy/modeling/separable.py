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
        array([[ True,  True,  True,  True], [ True,  True,  True,  True]]...)

    """
    return _separable(transform)


def _compute_n_outputs(left, right):
    """
    Compute the number of outputs of two models.
    The two models are the left and right model to an operation in
    the expression tree of a compound model.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    """
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


def _compute_n_inputs(left, right):
    """
    Compute the number of inputs of two models.
    The two models are the left and right model to an operation in
    the expression tree of a compound model.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    """
    if isinstance(left, Model):
        lnin = left.n_inputs
    else:
        lnin = left.shape[1]
    if isinstance(right, Model):
        rnin = right.n_inputs
    else:
        rnin = right.shape[1]
    ninp = lnin + rnin
    return ninp


def _arith_oper(left, right):
    """
    Function corresponding to one of the arithmetic operators
    ['+', '-'. '*', '/', '**'].
    This always returns a nonseparable output.


    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """
    # models have the same n_inputs and n_outputs
    if isinstance(left, Model):
        ninp = left.n_inputs
        noutp = left.n_outputs
    else:
        ninp, noutp = left.shape[1], left.shape[0]

    result = np.ones((noutp, ninp))
    return result


def _coord_matrix(model, pos, noutp, ninp):
    """
    Create an array representing inputs and outputs of a simple model.

    The array has a shape (noutp, ninp).
    Zeros indicate no correlation between an input and an output.
    Ones indicate correlation.

    Parameters
    ----------
    model : `astropy.modeling.Model`
        A simple model.
    pos : str
        Position of this model in the expression tree.
        One of ['left', 'right'].
    noutp : int
        Number of outputs of the compound model of which the input model
        is a left or right child.
    ninp : int
        Number of inputs of the compound model of which the input model
        is a left or right child.

    Returns
    -------
    result : ndarray
    """
    if isinstance(model, Mapping):
        axes = model.mapping
    else:
        axes = list(range(model.n_inputs))

    mat = np.zeros((noutp, ninp))
    if pos == 'left':
        mat[:model.n_outputs, :model.n_inputs] = 1
    else:
        mat[-model.n_outputs:, -model.n_inputs:] = 1

    if isinstance(model, Mapping):
        mat = mat[axes]

    return mat


def _cstack(left, right):
    """
    Function corresponding to '&' operation.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.

    """
    noutp = _compute_n_outputs(left, right)
    ninp = _compute_n_inputs(left, right)

    if isinstance(left, Model):
        cleft = _coord_matrix(left, 'left', noutp, ninp)
    else:
        cleft = np.zeros((noutp, ninp))
        cleft[: left.shape[0], : left.shape[1]] = left

    if isinstance(right, Model):
        cright = _coord_matrix(right, 'right', noutp, ninp)
    else:
        cright = np.zeros((noutp, ninp))
        cright[-right.shape[0]:, -right.shape[1]:] = right

    return cleft + cright


def _cdot(left, right):
    """
    Function corresponding to "|" operation.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """
    if isinstance(left, Model):
        cleft = _separable(left)
    else:
        cleft = left

    if isinstance(right, Model):
        cright = _separable(right)
    else:
        cright = right

    return np.dot(cleft, cright)


def _separable(transform):
    """
    Calculate the separability of outputs.

    Parameters
    ----------
    transform : `astropy.modeling.Model`
        A transform (usually a compound model).

    Returns
    -------
    separable_matrix : ndarray
        An array of shape (n_outputs, n_inputs).
    """
    if (transform_matrix := getattr(transform, '_separable_matrix', None)) is not None:
        return transform_matrix
    if isinstance(transform, CompoundModel):
        sepleft = _separable(transform.left)
        sepright = _separable(transform.right)
        return transform.op(sepleft, sepright)
    elif isinstance(transform, Model):
        return _coord_matrix(transform, 'left', transform.n_outputs, transform.n_inputs)
    else:
        # Handle the case where transform is already a matrix
        return transform


# Maps modeling operators to a function computing and represents the
# relationship between inputs and outputs.
_operators = {'&': _cstack, '|': _cdot, '+': _arith_oper, '-': _arith_oper,
              '*': _arith_oper, '/': _arith_oper, '**': _arith_oper}


CompoundModel.op = property(lambda self: _operators[self.op])
