# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from copy import deepcopy

import numpy as np

from astropy import log
from astropy.nddata.nduncertainty import (
    NDUncertainty,
    UnknownUncertainty,
    IncompatibleUncertaintiesWarning,
)
from astropy.units import dimensionless_unscaled
from astropy.utils import format_doc
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['NDArithmeticMixin']


# Global so it doesn't pollute the class dict unnecessarily:

# Docstring templates for add, subtract, multiply, divide methods.
_arit_doc = """
    Performs {name} by evaluating ``self`` {op} ``operand``.

    Parameters
    ----------
    operand : `NDData`-like instance
        If it is a `NDData`-like instance, the following attributes are used:
        ``data``, ``mask``, ``unit`` and ``uncertainty``.
        If it is not a `NDData`-like instance, only ``data`` and ``unit``
        attributes must be present. See "Notes" for additional information.

    propagate_uncertainties : `bool` or ``None``, optional
        If ``None`` the result will have no uncertainty. If ``False`` the
        result will have a copied version of the first operand that has an
        uncertainty. If ``True`` the result will have a correctly propagated
        uncertainty from the uncertainties of the operands but this assumes
        that the uncertainties are `NDUncertainty`-like. Default is ``True``.

        .. versionchanged:: 1.2
            This parameter must be given as keyword-parameter. Using it as
            positional parameter is deprecated.
            ``None`` was added as valid parameter value.

    handle_mask : callable, ``'first_found'`` or ``None``, optional
        If ``None`` the result will have no mask. If ``'first_found'`` the
        result will have a copied version of the first operand that has a
        mask). If it is a callable then the specified callable must
        create the results ``mask`` and if necessary provide a copy.
        Default is `numpy.logical_or`.

        .. versionadded:: 1.2

    handle_meta : callable, ``'first_found'`` or ``None``, optional
        If ``None`` the result will have no meta. If ``'first_found'`` the
        result will have a copied version of the first operand that has a
        (not empty) meta. If it is a callable then the specified callable must
        create the results ``meta`` and if necessary provide a copy.
        Default is ``None``.

        .. versionadded:: 1.2

    compare_wcs : callable, ``'first_found'`` or ``None``, optional
        If ``None`` the result will have no wcs and no comparison between
        the wcs of the operands is made. If ``'first_found'`` the
        result will have a copied version of the first operand that has a
        wcs. If it is a callable then the specified callable must
        compare the ``wcs``. The resulting ``wcs`` will be like the first
        operand that has a wcs (if one exists). See "Notes" for additional
        information.
        Default is ``'first_found'``.

        .. versionadded:: 1.2

    uncertainty_correlation : number or `~numpy.ndarray`, optional
        The correlation between the two operands is used for correct error
        propagation for correlated data. Default is 0.

        .. versionadded:: 1.2

    kwargs :
        Any other parameter that should be passed to the callables used.

    Returns
    -------
    result : `~astropy.nddata.NDData`-like
        The resulting dataset

    Notes
    -----
    If a ``callable`` is used for ``handle_mask``, ``handle_meta`` or
    ``compare_wcs`` the ``callable`` must accept the corresponding attributes
    as first two parameters. If the callable also needs additional parameters
    these can be defined as ``kwargs`` and must start with the same name as
    the callable followed by an underscore (``"_"``). For example
    ``handle_mask`` requires two parameters: ``mask1`` and ``mask2``, so
    ``handle_mask_kwarg1`` should be used for additional arguments.
"""

_arit_doc_returns = """
    Returns
    -------
    result : `~astropy.nddata.NDData`-like
        The resulting dataset
"""


class NDArithmeticMixin:
    """
    Mixin class to add arithmetic to an NDData object.

    When subclassing, be sure to list the superclasses in the correct order
    so that the subclass sees NDData as the main superclass. See
    `~astropy.nddata.NDDataRef` for an example.

    Notes
    -----
    This class only aims at covering the most common cases so there are certain
    restrictions on the saved attributes::

        - ``uncertainty`` : has to be something that has a `NDUncertainty`-like
          interface for uncertainty propagation
        - ``mask`` : has to be something that can be used by a bitwise ``or``
          operation.
        - ``wcs`` : has to implement a way of comparing with ``=`` to allow
          the operation.

    But there may be workarounds to allow arithmetic also on objects that
    don't meet these requirements. See the "+" operator for examples.

    Warning
    -------
    This class should never be instantiated directly. The ``NDData``
    arithmetic functions cannot work with a bare `NDArithmeticMixin` class.
    """

    def _arithmetic(self, operation, operand, propagate_uncertainties=True,
                    handle_mask=np.logical_or, handle_meta=None,
                    uncertainty_correlation=0, compare_wcs='first_found',
                    **kwds):
        """
        Base method which calculates the result of the arithmetic operation.

        This method determines the result of the arithmetic operation on the
        ``data`` including their units and then forwards to other methods
        to calculate the other attributes.

        Parameters
        ----------
        operation : callable
            The operation that is performed on the `NDData`. Supported are
            `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide`.

        operand : same type (class) as self
            see `NDArithmeticMixin.add`

        propagate_uncertainties : `bool` or ``None``, optional
            see `NDArithmeticMixin.add`

        handle_mask : callable, ``'first_found'`` or ``None``, optional
            see `NDArithmeticMixin.add`

        handle_meta : callable, ``'first_found'`` or ``None``, optional
            see `NDArithmeticMixin.add`

        compare_wcs : callable, ``'first_found'`` or ``None``, optional
            see `NDArithmeticMixin.add`

        uncertainty_correlation : ``Number`` or `~numpy.ndarray`, optional
            see `NDArithmeticMixin.add`

        kwargs :
            Any other parameter that should be passed to the different
            ``handle_*`` methods.

        Returns
        -------
        result : ndarray or `~astropy.units.Quantity`
            The resulting data as array (in case both operands were without
            unit) or as quantity if at least one had a unit.

        kwargs : `dict`
            The kwargs should contain all the other attributes (besides data
            and unit) needed to create a new instance for the result. This
            will be passed as ``**kwargs`` to the class ``__init__``.
        """
        # Find the appropriate keywords for the appropriate method (not sure
        # if data and uncertainty are ever used ...)
        kwds2 = {"mask": {}, "meta": {}, "wcs": {}, "data": {},
                 "uncertainty": {}}
        for i in kwds:
            splitted = i.split('_', 1)
            try:
                kwds2[splitted[0]][splitted[1]] = kwds[i]
            except KeyError:
                raise KeyError("Unknown prefix {0} for parameter {1}"
                               "".format(splitted[0], i))

        kwargs = {}
        # Do the calculation with data and unit
        # Returns what the data and unit should be in the result.
        result, init_kwds = self._arithmetic_data(operation, operand, **kwds2['data'])
        kwargs.update(init_kwds)

        # Determine the other properties
        if propagate_uncertainties is None:
            kwargs['uncertainty'] = None
        elif not propagate_uncertainties:
            if self.uncertainty is None:
                kwargs['uncertainty'] = deepcopy(operand.uncertainty)
            else:
                kwargs['uncertainty'] = deepcopy(self.uncertainty)
        else:
            kwargs['uncertainty'] = self._arithmetic_uncertainty(
                operation, operand, result, uncertainty_correlation,
                **kwds2['uncertainty'])

        # If both masks are None, don't calculate a mask.
        if self.mask is None and operand.mask is None:
            kwargs['mask'] = None
        elif handle_mask is None:
            kwargs['mask'] = None
        elif handle_mask == 'first_found':
            if self.mask is None:
                kwargs['mask'] = deepcopy(operand.mask)
            else:
                kwargs['mask'] = deepcopy(self.mask)
        else:
            # Handle the case where one mask is None but handle_mask is a callable
            if self.mask is None and operand.mask is not None:
                # If self.mask is None, create a False mask of the same shape as operand.mask
                self_mask = np.zeros_like(operand.mask, dtype=bool)
                kwargs['mask'] = handle_mask(self_mask, operand.mask, **kwds2['mask'])
            elif self.mask is not None and operand.mask is None:
                # If operand.mask is None, create a False mask of the same shape as self.mask
                operand_mask = np.zeros_like(self.mask, dtype=bool)
                kwargs['mask'] = handle_mask(self.mask, operand_mask, **kwds2['mask'])
            else:
                # Both masks exist, use them directly
                kwargs['mask'] = handle_mask(self.mask, operand.mask, **kwds2['mask'])

        if compare_wcs is None:
            kwargs['wcs'] = None
        elif compare_wcs == 'first_found':
            if self.wcs is None and hasattr(operand, 'wcs'):
                kwargs['wcs'] = deepcopy(operand.wcs)
            else:
                kwargs['wcs'] = deepcopy(self.wcs)
        else:
            kwargs['wcs'] = self._arithmetic_wcs(operation, operand,
                                                  compare_wcs, **kwds2['wcs'])

        if handle_meta is None:
            kwargs['meta'] = None
        elif handle_meta == 'first_found':
            if not self.meta:
                kwargs['meta'] = deepcopy(operand.meta)
            else:
                kwargs['meta'] = deepcopy(self.meta)
        else:
            kwargs['meta'] = handle_meta(self.meta, operand.meta,
                                         **kwds2['meta'])

        return result, kwargs

    def _arithmetic_data(self, operation, operand, **kwds):
        """
        Calculate the resulting data

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin.add`

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        kwds :
            Additional parameters.

        Returns
        -------
        result_data : ndarray or `~astropy.units.Quantity`
            The resulting data as array (in case both operands were without
            unit) or as quantity if at least one had a unit.

        result_unit : `~astropy.units.Unit`
            The resulting unit
        """
        # Do the calculation with data and unit
        # Returns what the data and unit should be in the result.
        if hasattr(operand, 'unit') and operand.unit is not None:
            result_unit = operand.unit
        else:
            result_unit = None

        if hasattr(self, 'unit') and self.unit is not None:
            if result_unit is not None:
                # Both have units so calculate the resulting unit and convert
                # the operand data if needed.
                if operation in (np.multiply, np.true_divide, np.divide,
                                 np.floor_divide):
                    if operation in (np.multiply,):
                        result_unit = self.unit * result_unit
                    elif operation in (np.true_divide, np.divide,
                                       np.floor_divide):
                        result_unit = self.unit / result_unit
                else:
                    # Addition or subtraction: Create a Quantity to let
                    # astropy.units determine the result instead of trying to
                    # determine it manually.
                    try:
                        result_unit = (self.unit * 1 + operand.unit * 1).unit
                    except Exception:
                        # Let units determine what to do if the units do not
                        # match
                        result_unit = (self.unit * 1 - operand.unit * 1).unit
                result_data = operation(self.data, operand.data)
            else:
                # Only self has a unit
                result_unit = self.unit
                result_data = operation(self.data, operand.data)
        else:
            if result_unit is not None:
                # Only operand has a unit.
                result_data = operation(self.data, operand.data)
            else:
                # Neither has a unit.
                result_unit = None
                result_data = operation(self.data, operand.data)

        return result_data, {'unit': result_unit}

    def _arithmetic_uncertainty(self, operation, operand, result, correlation,
                                **kwds):
        """
        Calculate the resulting uncertainty.

        This method can be overridden if one wants to implement a different
        uncertainty propagation.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin.add`

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        result : `~astropy.units.Quantity` or `~numpy.ndarray`
            The result of `self.data operation operand.data`.
            This parameter is not used by default but may be used by
            subclasses. By default the result is calculated again.

        correlation : number or `~numpy.ndarray`
            see `NDArithmeticMixin.add`

        kwds :
            Additional parameters.

        Returns
        -------
        result_uncertainty : `NDUncertainty` subclass instance or None
            The resulting uncertainty already saved in the same `NDUncertainty`
            subclass that ``self`` had (or ``operand`` if self had no
            uncertainty). ``None`` only if both had no uncertainty.
        """
        # Make sure these uncertainties are NDUncertainties so this kind of
        # propagation is possible.
        if self.uncertainty is not None and not isinstance(
                self.uncertainty, NDUncertainty):
            warnings.warn("Uncertainty propagation is not supported for "
                          "{0} uncertainties. The uncertainty will be "
                          "ignored.".format(type(self.uncertainty)),
                          IncompatibleUncertaintiesWarning)
            self_uncertainty = None
        else:
            self_uncertainty = self.uncertainty

        if hasattr(operand, 'uncertainty') and operand.uncertainty is not None:
            if not isinstance(operand.uncertainty, NDUncertainty):
                warnings.warn("Uncertainty propagation is not supported for "
                              "{0} uncertainties. The uncertainty will be "
                              "ignored.".format(type(operand.uncertainty)),
                              IncompatibleUncertaintiesWarning)
                operand_uncertainty = None
            else:
                operand_uncertainty = operand.uncertainty
        else:
            operand_uncertainty = None

        # Now do the uncertainty propagation
        # TODO: There is no enforced requirement that actually forbids the
        # uncertainty to have negative entries but with correlation the
        # sign of the uncertainty DOES matter.
        if self_uncertainty is None and operand_uncertainty is None:
            # Neither has uncertainties
            result_uncertainty = None

        elif self_uncertainty is None:
            # Take the operand uncertainty
            result_uncertainty = operand_uncertainty._propagate_add(
                operand, result, correlation)

        elif operand_uncertainty is None:
            # Take the self uncertainty
            result_uncertainty = self_uncertainty._propagate_add(
                self, result, correlation)

        else:
            # Both have uncertainties
            # Check if the uncertainties are the same class in order to do
            # propagation...
            if self_uncertainty.__class__ != operand_uncertainty.__class__:
                warnings.warn("Propagation of uncertainties of different "
                              "classes is not supported: "
                              "{0} vs {1}. Uncertainties will be ignored."
                              "".format(self_uncertainty.__class__.__name__,
                                        operand_uncertainty.__class__.__name__),
                              IncompatibleUncertaintiesWarning)
                result_uncertainty = None
            else:
                # Determine which propagation method should be used.
                if operation in (np.add, np.subtract):
                    propagate = self_uncertainty._propagate_add
                elif operation in (np.multiply, np.true_divide, np.divide,
                                   np.floor_divide):
                    propagate = self_uncertainty._propagate_multiply
                else:
                    result_uncertainty = None

                if result_uncertainty is not None:
                    result_uncertainty = propagate(operand_uncertainty, result,
                                                    correlation)

        return result_uncertainty

    def _arithmetic_wcs(self, operation, operand, compare_wcs, **kwds):
        """
        Calculate the resulting wcs.

        There is actually no calculation involved but it is a good place to
        compare wcs information of both operands. This method is called only
        if ``compare_wcs`` is a callable.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin.add`

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        compare_wcs : callable
            see `NDArithmeticMixin.add`

        kwds :
            Additional parameters.

        Returns
        -------
        result_wcs : any type
            The resulting wcs.
        """
        # ok, not really arithmetic but we need to check which wcs makes sense
        # for the result and this is an ideal place to do it.
        if self.wcs is None and operand.wcs is None:
            return None
        elif self.wcs is None:
            return deepcopy(operand.wcs)
        elif operand.wcs is None:
            return deepcopy(self.wcs)
        else:
            compare_wcs(self.wcs, operand.wcs, **kwds)
            return deepcopy(self.wcs)

    @format_doc(_arit_doc, name="addition", op="+")
    def add(self, operand, **kwargs):
        return self._arithmetic(np.add, operand, **kwargs)

    @format_doc(_arit_doc, name="subtraction", op="-")
    def subtract(self, operand, **kwargs):
        return self._arithmetic(np.subtract, operand, **kwargs)

    @format_doc(_arit_doc, name="multiplication", op="*")
    def multiply(self, operand, **kwargs):
        return self._arithmetic(np.multiply, operand, **kwargs)

    @format_doc(_arit_doc, name="division", op="/")
    def divide(self, operand, **kwargs):
        return self._arithmetic(np.true_divide, operand, **kwargs)

    def __add__(self, operand):
        return self.add(operand)

    def __radd__(self, operand):
        return self.add(operand)

    def __sub__(self, operand):
        return self.subtract(operand)

    def __rsub__(self, operand):
        # Subtract self from operand
        return self.__class__(operand).subtract(self)

    def __mul__(self, operand):
        return self.multiply(operand)

    def __rmul__(self, operand):
        return self.multiply(operand)

    def __truediv__(self, operand):
        return self.divide(operand)

    def __rtruediv__(self, operand):
        # Divide operand by self
        return self.__class__(operand).divide(self)

    def __pow__(self, operand):
        return self._arithmetic(np.power, operand)

    def __rpow__(self, operand):
        return self.__class__(operand).__pow__(self)
