# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the Arithmetic mixin to the NDData class.

import warnings
from copy import deepcopy

import numpy as np

from astropy.nddata.nduncertainty import NDUncertainty
from astropy.units import dimensionless_unscaled
from astropy.utils import format_doc, sharedmethod
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import Masked

__all__ = ["NDArithmeticMixin"]

# Global so it doesn't pollute the class dict unnecessarily:

# Docstring templates for add, subtract, multiply, divide methods.
_arit_doc = """
    Performs {name} by evaluating ``self`` {op} ``operand``.

    Parameters
    ----------
    operand, operand2 : `NDData`-like instance
        If ``operand2`` is ``None`` or not given it will perform the operation
        ``self`` {op} ``operand``.
        If ``operand2`` is given it will perform ``operand`` {op} ``operand2``.
        If the method was called on a class rather than on the instance
        ``operand2`` must be given.

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
        compare the ``wcs``. The resulting ``wcs`` will be like if ``False``
        was given, ``None`` or no ``wcs`` at all.
        Default is ``'first_found'``.

        .. versionadded:: 1.2

    uncertainty_correlation : number or `~numpy.ndarray`, optional
        The correlation between the two operands is used for correct error
        propagation for correlated data as given in:
        https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Non-linear_combinations
        Default is 0.

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
    these can be given as ``kwargs``.

    ``handle_mask`` must accept 2 parameters:

    - ``mask1, mask2`` : `numpy.ndarray` or ``None``
        The masks (if any) of the two operands that should be combined.

    ``handle_meta`` must accept 2 parameters:

    - ``meta1, meta2`` : ``dict``
        The meta dictionaries (if any) of the two operands that should be
        combined.

    ``compare_wcs`` must accept 2 parameters:

    - ``wcs1, wcs2`` : ``None`` or `~astropy.wcs.WCS`-like
        The wcs of the two operands that should be compared.

    Examples
    --------
    Using this mixin:

        >>> from astropy.nddata import NDData, StdDevUncertainty
        >>> nd1 = NDData([1,1,1], uncertainty=StdDevUncertainty([0.1, 0.1, 0.1]))
        >>> nd2 = NDData([1,1,1], uncertainty=StdDevUncertainty([0.2, 0.2, 0.2]))
        >>> nd1.{name}(nd2) # doctest: +SKIP
        NDData([...])
    """


def _arithmetic_mask(mask1, mask2, handle_mask, axis=None, **kwds):
    """
    Calculate the resulting mask.

    This is a private function used in `NDArithmeticMixin._arithmetic` method.
    """
    # If only one mask is present, we need to handle it appropriately
    if mask1 is None and mask2 is None:
        return None
    elif mask1 is None:
        if handle_mask is None:
            return None
        elif handle_mask == "first_found":
            if mask2 is None:
                return None
            else:
                return deepcopy(mask2)
        else:
            # For callable handle_mask, we need to handle None masks
            # Convert None to appropriate array of False values
            if mask2 is not None:
                mask1_converted = np.zeros_like(mask2, dtype=bool)
                return handle_mask(mask1_converted, mask2, **kwds)
            else:
                return None
    elif mask2 is None:
        if handle_mask is None:
            return None
        elif handle_mask == "first_found":
            return deepcopy(mask1)
        else:
            # For callable handle_mask, we need to handle None masks
            # Convert None to appropriate array of False values
            mask2_converted = np.zeros_like(mask1, dtype=bool)
            return handle_mask(mask1, mask2_converted, **kwds)
    else:
        # Both masks are present
        if handle_mask is None:
            return None
        elif handle_mask == "first_found":
            return deepcopy(mask1)
        else:
            return handle_mask(mask1, mask2, **kwds)


def _arithmetic_uncertainty(uncertainty1, uncertainty2, data1, data2,
                            result_data, propagate_uncertainties, result_unit,
                            correlation, operation, **kwds):
    """
    Calculate the resulting uncertainty.

    This is a private function used in `NDArithmeticMixin._arithmetic` method.
    """
    # Make sure these uncertainties are NDUncertainties so this kind of
    # propagation is possible.
    if propagate_uncertainties is None:
        result_uncertainty = None

    elif not propagate_uncertainties:
        if uncertainty1 is not None:
            result_uncertainty = uncertainty1
        else:
            result_uncertainty = uncertainty2
    else:
        if uncertainty1 is None and uncertainty2 is None:
            result_uncertainty = None
        elif (
            isinstance(uncertainty1, NDUncertainty)
            and isinstance(uncertainty2, NDUncertainty)
        ):
            result_uncertainty = uncertainty1.propagate(
                operation, uncertainty2, result_data, correlation
            )
        elif uncertainty1 is not None:
            result_uncertainty = uncertainty1.propagate(
                operation, uncertainty2, result_data, correlation
            )
        elif uncertainty2 is not None:
            result_uncertainty = uncertainty2.propagate(
                operation, uncertainty1, result_data, correlation
            )
        else:
            result_uncertainty = None

    return result_uncertainty


def _arithmetic_wcs(wcs1, wcs2, compare_wcs, **kwds):
    """
    Calculate the resulting wcs.

    This is a private function used in `NDArithmeticMixin._arithmetic` method.
    """
    if compare_wcs is None:
        return None
    elif compare_wcs == "first_found":
        if wcs1 is None:
            return deepcopy(wcs2)
        else:
            return deepcopy(wcs1)
    else:
        # Let the callable handle it but give it some keyword arguments.
        # If the callable returns False then return None instead
        wcs_compare = compare_wcs(wcs1, wcs2, **kwds)
        if wcs_compare is False:
            return None
        else:
            return wcs_compare


def _arithmetic_meta(meta1, meta2, handle_meta, **kwds):
    """
    Calculate the resulting meta.

    This is a private function used in `NDArithmeticMixin._arithmetic` method.
    """
    if handle_meta is None:
        return None
    elif handle_meta == "first_found":
        if meta1:
            return deepcopy(meta1)
        else:
            return deepcopy(meta2)
    else:
        # Let the callable handle it.
        return handle_meta(meta1, meta2, **kwds)


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
          the operation only or raise an Exception if the ``wcs`` do not match.

    But there is a workaround that allows to disable handling a specific
    attribute and to simply set the results attribute to ``None`` or to copy
    the existing attribute (and neglecting the other).

    For several methods, a ``handle_mask`` parameter is available that accepts
    a callable to create the output mask. The default of `numpy.logical_or`
    means that the masks are combined with the logical "or" operation.
    However, for a bitwise "or" operation, the parameter can be set to
    `numpy.bitwise_or`.

    .. warning::
        Only the ``mask`` and ``wcs`` attributes are compared between operands.
        The ``data`` itself is **NOT** compared.

        .. versionadded:: 1.2
    """

    def _arithmetic(
        self,
        operation,
        operand,
        propagate_uncertainties=True,
        handle_mask=np.logical_or,
        handle_meta=None,
        uncertainty_correlation=0,
        compare_wcs="first_found",
        operation_ignores_mask=False,
        axis=None,
        **kwds
    ):
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
            see :meth:`NDArithmeticMixin.add`

        propagate_uncertainties : `bool` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`

        handle_mask : callable, ``'first_found'`` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`

        handle_meta : callable, ``'first_found'`` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`

        uncertainty_correlation : ``Number`` or `~numpy.ndarray`, optional
            see :meth:`NDArithmeticMixin.add`

        compare_wcs : callable, ``'first_found'`` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`

        operation_ignores_mask : bool, optional
            When `True`, masked values are ignored during arithmetic.
            Default is `False`.

        axis : int, optional
            axis along which the operation should be performed.

        kwargs :
            Any other parameter that should be passed to the different
            ``handle_*`` functions.

        Returns
        -------
        result : ndarray or `~astropy.units.Quantity`
            The resulting data as array (in case both operands were without
            unit) or as quantity if at least one had a unit.

        kwargs : `dict`
            The kwargs should contain all the other attributes (besides data)
            needed to create a new instance for the result. Creating the new
            instance is NOT part of this method.
        """
        # Find the appropriate keywords for the result and do the arithmetic.
        # This method is not responsible for creating the result instance
        # because the result should have the same class as the method was
        # called on.
        if isinstance(operand, self.__class__):
            # Let's extract the data and if there are units save them for later
            # use.
            if hasattr(self, "unit") and hasattr(operand, "unit"):
                if self.unit is None:
                    self_data = self.data
                else:
                    self_data = self.data << self.unit
                if operand.unit is None:
                    operand_data = operand.data
                else:
                    operand_data = operand.data << operand.unit
            else:
                self_data = self.data
                operand_data = operand.data

            # preserve original units
            if hasattr(self, "unit") or hasattr(operand, "unit"):
                if not hasattr(self, "unit"):
                    self_unit = dimensionless_unscaled
                else:
                    self_unit = self.unit
                if not hasattr(operand, "unit"):
                    operand_unit = dimensionless_unscaled
                else:
                    operand_unit = operand.unit
            else:
                self_unit = None
                operand_unit = None

            # Make the operation with the data
            result_data = operation(self_data, operand_data)

            # Determine the unit of the result
            if self_unit is None and operand_unit is None:
                result_unit = None
            elif self_unit is None:
                result_unit = operand_unit
            elif operand_unit is None:
                result_unit = self_unit
            else:
                result_unit = operation(self_unit, operand_unit)

            # Determine the other properties
            if hasattr(self, "mask") or hasattr(operand, "mask"):
                if not hasattr(self, "mask"):
                    self_mask = None
                else:
                    self_mask = self.mask
                if not hasattr(operand, "mask"):
                    operand_mask = None
                else:
                    operand_mask = operand.mask
            else:
                self_mask = None
                operand_mask = None

            result_mask = _arithmetic_mask(
                self_mask, operand_mask, handle_mask, axis=axis, **kwds
            )

            if hasattr(self, "uncertainty") or hasattr(operand, "uncertainty"):
                if not hasattr(self, "uncertainty"):
                    self_uncertainty = None
                else:
                    self_uncertainty = self.uncertainty
                if not hasattr(operand, "uncertainty"):
                    operand_uncertainty = None
                else:
                    operand_uncertainty = operand.uncertainty
            else:
                self_uncertainty = None
                operand_uncertainty = None

            result_uncertainty = _arithmetic_uncertainty(
                self_uncertainty,
                operand_uncertainty,
                self_data,
                operand_data,
                result_data,
                propagate_uncertainties,
                result_unit,
                uncertainty_correlation,
                operation,
                **kwds
            )

            # The wcs framework is not yet implemented, just compare WCS
            if hasattr(self, "wcs") or hasattr(operand, "wcs"):
                if not hasattr(self, "wcs"):
                    self_wcs = None
                else:
                    self_wcs = self.wcs
                if not hasattr(operand, "wcs"):
                    operand_wcs = None
                else:
                    operand_wcs = operand.wcs
            else:
                self_wcs = None
                operand_wcs = None

            result_wcs = _arithmetic_wcs(self_wcs, operand_wcs, compare_wcs, **kwds)

            # ok, determine the meta
            if hasattr(self, "meta") or hasattr(operand, "meta"):
                if not hasattr(self, "meta"):
                    self_meta = {}
                else:
                    self_meta = self.meta
                if not hasattr(operand, "meta"):
                    operand_meta = {}
                else:
                    operand_meta = operand.meta
            else:
                self_meta = {}
                operand_meta = {}

            result_meta = _arithmetic_meta(self_meta, operand_meta, handle_meta, **kwds)

        else:
            # Let's assume that the operand is a number or something that can
            # be used with the data.
            result_data = operation(self.data, operand)

            if hasattr(self, "unit"):
                result_unit = operation(self.unit, 1)
            else:
                result_unit = None

            if hasattr(self, "mask"):
                # Masks are copied and not arithmetically combined.
                result_mask = deepcopy(self.mask)
            else:
                result_mask = None

            if hasattr(self, "uncertainty") and propagate_uncertainties:
                # Create an uncertainty for the operand
                result_uncertainty = self.uncertainty.propagate(
                    operation, operand, result_data, uncertainty_correlation
                )
            elif hasattr(self, "uncertainty") and not propagate_uncertainties:
                result_uncertainty = deepcopy(self.uncertainty)
            else:
                result_uncertainty = None

            if hasattr(self, "wcs"):
                # Just copy the WCS
                result_wcs = deepcopy(self.wcs)
            else:
                result_wcs = None

            if hasattr(self, "meta"):
                # Just copy the meta
                result_meta = deepcopy(self.meta)
            else:
                result_meta = None

        # Wrap the individual results into a dict and return it.
        result_kwds = {}
        result_kwds["data"] = result_data
        if result_unit is not None:
            result_kwds["unit"] = result_unit
        if result_uncertainty is not None:
            result_kwds["uncertainty"] = result_uncertainty
        if result_mask is not None:
            result_kwds["mask"] = result_mask
        if result_wcs is not None:
            result_kwds["wcs"] = result_wcs
        if result_meta is not None:
            result_kwds["meta"] = result_meta

        return result_kwds

    @sharedmethod
    @format_doc(_arit_doc, name="addition", op="+")
    def add(self, operand, operand2=None, **kwargs):
        return self._prepare_then_do_arithmetic(np.add, operand, operand2, **kwargs)

    @sharedmethod
    @format_doc(_arit_doc, name="subtraction", op="-")
    def subtract(self, operand, operand2=None, **kwargs):
        return self._prepare_then_do_arithmetic(
            np.subtract, operand, operand2, **kwargs
        )

    @sharedmethod
    @format_doc(_arit_doc, name="multiplication", op="*")
    def multiply(self, operand, operand2=None, **kwargs):
        return self._prepare_then_do_arithmetic(
            np.multiply, operand, operand2, **kwargs
        )

    @sharedmethod
    @format_doc(_arit_doc, name="division", op="/")
    def divide(self, operand, operand2=None, **kwargs):
        return self._prepare_then_do_arithmetic(
            np.true_divide, operand, operand2, **kwargs
        )

    def _prepare_then_do_arithmetic(self, operation, operand, operand2, **kwargs):
        """
        Intermediate method called by public arithmetic methods after
        operand validation.
        """
        if operand2 is None:
            return self._do_arithmetic(operation, operand, **kwargs)
        else:
            return operand._do_arithmetic(operation, operand2, **kwargs)

    def _do_arithmetic(self, operation, operand, **kwargs):
        """
        {name} another dataset (`operand`) to this dataset.

        Parameters
        ----------
        operand : `NDData`-like instance
            The second operand in the operation a {op} b
        kwargs :
            Additional parameters given to :meth:`NDArithmeticMixin._arithmetic`.

        Returns
        -------
        result : `~astropy.nddata.NDData`-like
            The resulting dataset
        """
        # DO the arithmetic operation
        kwargs = self._arithmetic(operation, operand, **kwargs)
        # Return a new class
        return self.__class__(**kwargs)

    def __add__(self, operand):
        return self.add(operand)

    def __radd__(self, operand):
        return self.add(operand)

    def __sub__(self, operand):
        return self.subtract(operand)

    def __rsub__(self, operand):
        # Subtract self from operand
        if hasattr(operand, "subtract"):
            return operand.subtract(self)
        else:
            # operand doesn't know how to subtract NDData, so we do it here
            return self.__class__(operand).subtract(self)

    def __mul__(self, operand):
        return self.multiply(operand)

    def __rmul__(self, operand):
        return self.multiply(operand)

    def __truediv__(self, operand):
        return self.divide(operand)

    def __rtruediv__(self, operand):
        # Divide operand by self
        if hasattr(operand, "divide"):
            return operand.divide(self)
        else:
            # operand doesn't know how to divide NDData, so we do it here
            return self.__class__(operand).divide(self)

    def __pow__(self, operand):
        return self._do_arithmetic(np.power, operand)

    def __rpow__(self, operand):
        if hasattr(operand, "__pow__"):
            return operand.__pow__(self)
        else:
            return self.__class__(operand).__pow__(self)

    def __iadd__(self, operand):
        return self.add(operand)

    def __isub__(self, operand):
        return self.subtract(operand)

    def __imul__(self, operand):
        return self.multiply(operand)

    def __itruediv__(self, operand):
        return self.divide(operand)

    def __ipow__(self, operand):
        return self.__pow__(operand)

    def __neg__(self):
        # arithmetic with 0 - self
        return self.multiply(-1)

    def __pos__(self):
        return deepcopy(self)

    def __abs__(self):
        return self._do_arithmetic(np.absolute, None)
