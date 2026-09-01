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
        was given, ``None`` or no ``wcs`` is given.
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
    these can be defined as ``kwargs`` and must start with the same name as
    the callable followed by an underscore (``"_"``).

    Examples
    --------
    Using this mixin::

        >>> from astropy.nddata import NDData, StdDevUncertainty
        >>> nd1 = NDData([1,2,3], uncertainty=StdDevUncertainty([0.1, 0.1, 0.1]))
        >>> nd2 = NDData([1,2,3], uncertainty=StdDevUncertainty([0.1, 0.1, 0.1]))
        >>> nd1.{name}(nd2) # doctest: +SKIP
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

    But there is a workaround that allows to disable handling a specific
    attribute and to simply set the results attribute to ``None`` or to
    copy the existing attribute (and neglecting the other).
    For example for uncertainties not representing an `NDUncertainty`-like
    interface you can use ``propagate_uncertainties=None`` to disable
    uncertainty propagation.
    For custom mask handling you can provide a callable for ``handle_mask``.
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

        uncertainty_correlation : `float` or `~numpy.ndarray`, optional
            see :meth:`NDArithmeticMixin.add`

        compare_wcs : callable, ``'first_found'`` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`

        operation_ignores_mask : bool, optional
            When `True`, masked values are ignored during the operation.
            Default is `False`.

        axis : int or tuple of ints, optional
            axis or axes over which the operation should be performed.

        kwds :
            Any other parameter that should be passed to the different
            methods called in this method.

        Returns
        -------
        result : ndarray or `~astropy.units.Quantity`
            The resulting data as array (in case both operands were without
            unit) or as quantity if at least one had a unit.

        kwargs : `dict`
            The kwargs should contain all the other attributes (besides data
            and unit) needed to create a new instance for the result. Creating
            the new instance is *not* done in this method!
        """
        # Find the appropriate keywords for the appropriate method
        kwds2 = {"handle_mask": handle_mask, "handle_meta": handle_meta}
        kwds2.update(kwds)

        # First check that the WCS allows the arithmetic operation
        if compare_wcs is None:
            kwargs = {"wcs": None}
        elif compare_wcs in ("first_found", "ff"):
            if self.wcs is None and hasattr(operand, "wcs"):
                kwargs = {"wcs": deepcopy(operand.wcs)}
            else:
                kwargs = {"wcs": deepcopy(self.wcs)}
        else:
            kwargs = {"wcs": compare_wcs(self.wcs, operand.wcs, **kwds2)}

        # Then calculate the resulting data (which can but doesn't have to be a
        # quantity)
        result = self._arithmetic_data(operation, operand, **kwds2)
        # determine the other properties
        if propagate_uncertainties is None:
            kwargs["uncertainty"] = None
        elif not propagate_uncertainties:
            if self.uncertainty is None:
                kwargs["uncertainty"] = deepcopy(getattr(operand, "uncertainty", None))
            else:
                kwargs["uncertainty"] = deepcopy(self.uncertainty)
        else:
            kwargs["uncertainty"] = self._arithmetic_uncertainty(
                operation, operand, result, uncertainty_correlation, **kwds2
            )

        # Calculate the resulting mask
        kwargs["mask"] = self._arithmetic_mask(
            operation, operand, handle_mask, axis=axis, **kwds2
        )

        # Calculate the resulting meta
        if handle_meta is None:
            kwargs["meta"] = None
        elif handle_meta in ("first_found", "ff"):
            if not self.meta:
                kwargs["meta"] = deepcopy(getattr(operand, "meta", None))
            else:
                kwargs["meta"] = deepcopy(self.meta)
        else:
            kwargs["meta"] = handle_meta(self.meta, operand.meta, **kwds2)

        return result, kwargs

    def _arithmetic_data(self, operation, operand, **kwds):
        """
        Calculate the resulting data.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin._arithmetic` parameter description.

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        kwds :
            Additional parameters.

        Returns
        -------
        result : `~astropy.units.Quantity` or `~numpy.ndarray`
            The resulting data as quantity if at least one of the operands had
            a unit otherwise a normal numpy array.

        Notes
        -----
        If both operands have a ``unit`` the resulting data will have the
        appropriate unit. The unit itself is determined using the
        `~astropy.units` package. Also the unit is checked so that only
        "compatible" operations are allowed.
        """
        # Do the calculation with or without units
        if self.unit is None:
            if hasattr(operand, "unit") and operand.unit is not None:
                # If only the operand has a unit and a (compatible) unit is
                # needed for the result, convert the self's data to that unit.
                try:
                    scale_to_other_unit = dimensionless_unscaled.to(operand.unit)
                    result = operation(
                        scale_to_other_unit * self.data, operand.data
                    )
                except Exception:
                    result = operation(self.data, operand.data)
            else:
                result = operation(self.data, operand.data)
        elif hasattr(operand, "unit"):
            if operand.unit is not None:
                result = operation(self.data * self.unit, operand.data * operand.unit)
            else:
                # If only self has a unit and a (compatible) unit is needed for
                # the result, convert the operand to that unit.
                try:
                    scale_to_other_unit = dimensionless_unscaled.to(self.unit)
                    result = operation(
                        self.data, scale_to_other_unit * operand.data
                    )
                except Exception:
                    result = operation(self.data, operand.data)
        else:
            result = operation(self.data * self.unit, operand.data)

        return result

    def _arithmetic_uncertainty(self, operation, operand, result, correlation, **kwds):
        """
        Calculate the resulting uncertainty.

        This method can be overridden if one wants to implement a different
        uncertainty propagation.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin._arithmetic` parameter description.

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        result : `~astropy.units.Quantity` or `~numpy.ndarray`
            The result of `NDArithmeticMixin._arithmetic_data`.

        correlation : float or `~numpy.ndarray`
            see `NDArithmeticMixin._arithmetic` parameter description.

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
            self.uncertainty, NDUncertainty
        ):
            warnings.warn(
                "Uncertainty propagation is only available for "
                "subclasses of NDUncertainty.",
                AstropyUserWarning,
            )
            return None
        if (
            hasattr(operand, "uncertainty")
            and operand.uncertainty is not None
            and not isinstance(operand.uncertainty, NDUncertainty)
        ):
            warnings.warn(
                "Uncertainty propagation is only available for "
                "subclasses of NDUncertainty.",
                AstropyUserWarning,
            )
            return None

        # Now do the uncertainty propagation
        # TODO: There is no enforced requirement that actually forbids the
        # uncertainty to have negative entries but with correlation the
        # sign of the uncertainty DOES matter.
        if self.uncertainty is None and (
            not hasattr(operand, "uncertainty") or operand.uncertainty is None
        ):
            # Neither has uncertainties
            return None
        elif self.uncertainty is None:
            # Create a temporary uncertainty to allow uncertainty propagation
            # to yield the correct results. (issue #4152)
            self.uncertainty = operand.uncertainty.__class__(None)
            result_uncert = self.uncertainty.propagate(
                operation, operand, result, correlation
            )
            # Delete the temporary uncertainty again.
            self.uncertainty = None
            return result_uncert
        elif not hasattr(operand, "uncertainty") or operand.uncertainty is None:
            # Create a temporary uncertainty to allow uncertainty propagation
            # to yield the correct results. (issue #4152)
            operand.uncertainty = self.uncertainty.__class__(None)
            result_uncert = self.uncertainty.propagate(
                operation, operand, result, correlation
            )
            # Delete the temporary uncertainty again.
            operand.uncertainty = None
            return result_uncert
        else:
            # Both have uncertainties
            return self.uncertainty.propagate(operation, operand, result, correlation)

    def _arithmetic_mask(self, operation, operand, handle_mask, axis=None, **kwds):
        """
        Calculate the resulting mask.

        This is implemented as the piecewise ``or`` operation if both have a
        mask.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin._arithmetic` parameter description.
            By default, the ``operation`` will be ignored.

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        handle_mask : callable, ``'first_found'`` or ``None``
            see `NDArithmeticMixin._arithmetic` parameter description.
            If either ``self`` or ``operand`` have a mask then the returned
            mask will be like if ``'first_found'`` was given.

        kwds :
            Additional parameters given to ``handle_mask``.

        Returns
        -------
        result_mask : any type
            If ``None`` was given, ``None`` will be returned otherwise the
            result depends on the ``handle_mask`` parameter.
        """
        # If only one mask is present we need not bother about any type checks
        if (
            self.mask is None and operand is not None and hasattr(operand, "mask")
        ) and operand.mask is not None:
            # Make a copy so there is no reference in the result.
            return deepcopy(operand.mask)
        elif operand is None or not hasattr(operand, "mask") or operand.mask is None:
            if self.mask is None:
                return None
            else:
                return deepcopy(self.mask)
        elif self.mask is None and operand.mask is not None:
            # Make a copy so there is no reference in the result.
            return deepcopy(operand.mask)

        # Now lets calculate the resulting mask (self and operand both have
        # a mask)
        elif handle_mask is None:
            return None
        elif handle_mask in ("first_found", "ff"):
            if self.mask is None:
                return deepcopy(operand.mask)
            else:
                return deepcopy(self.mask)
        else:
            # Handle the case where one or both masks might be None
            if self.mask is None and operand.mask is None:
                return None
            elif self.mask is None:
                return deepcopy(operand.mask)
            elif operand.mask is None:
                return deepcopy(self.mask)
            else:
                # Both masks are not None, apply the handle_mask function
                return handle_mask(self.mask, operand.mask, **kwds)

    def _arithmetic_wcs(self, operation, operand, compare_wcs, **kwds):
        """
        Calculate the resulting wcs.

        There is actually no calculation involved but it is a good place to
        add a "sanity check" that the WCS allows the arithmetic operation.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin._arithmetic` parameter description.

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        compare_wcs : callable, ``'first_found'`` or ``None``
            see `NDArithmeticMixin._arithmetic` parameter description.

        kwds :
            Additional parameters given to ``compare_wcs``.

        Returns
        -------
        result_wcs : any type
            The resulting wcs.
        """
        # ok, not really arithmetic but we need to check which wcs makes sense
        # for the result and this is an ideal place to compare wcs.
        if compare_wcs is None:
            return None
        elif compare_wcs in ("first_found", "ff"):
            if self.wcs is None and hasattr(operand, "wcs"):
                return deepcopy(operand.wcs)
            else:
                return deepcopy(self.wcs)
        else:
            # Let the callable handle it but give it the wcs as arguments.
            return compare_wcs(self.wcs, operand.wcs, **kwds)

    def _prepare_then_do_arithmetic(self, other, cls, **kwargs):
        if other is None:
            return NotImplemented
        if not issubclass(cls, NDArithmeticMixin):
            return NotImplemented

        other_wrapped = cls(other)
        return self._arithmetic(other_wrapped, **kwargs)

    @sharedmethod
    @format_doc(_arit_doc, name="addition", op="+")
    def add(self, operand, operand2=None, **kwargs):
        return self._prepare_then_do_arithmetic(
            operand, operand2, np.add, **kwargs
        )

    @sharedmethod
    @format_doc(_arit_doc, name="subtraction", op="-")
    def subtract(self, operand, operand2=None, **kwargs):
        return self._prepare_then_do_arithmetic(
            operand, operand2, np.subtract, **kwargs
        )

    @sharedmethod
    @format_doc(_arit_doc, name="multiplication", op="*")
    def multiply(self, operand, operand2=None, **kwargs):
        return self._prepare_then_do_arithmetic(
            operand, operand2, np.multiply, **kwargs
        )

    @sharedmethod
    @format_doc(_arit_doc, name="division", op="/")
    def divide(self, operand, operand2=None, **kwargs):
        return self._prepare_then_do_arithmetic(
            operand, operand2, np.true_divide, **kwargs
        )

    def _prepare_then_do_arithmetic(self, operand, operand2, operation, **kwargs):
        """
        Boilerplate method which wraps the operands and calls ``_arithmetic``.

        Parameters
        ----------
        operand, operand2 : `NDData`-like or None
            see :meth:`NDArithmeticMixin.add`

        operation : callable
            The operation that is performed on the `NDData`. Supported are
            `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide`.

        kwargs :
            Additional "options" that can be given to the arithmetic operation.

        Returns
        -------
        result : `NDData`-like
            Results of the arithmetic operation.

        Notes
        -----
        If ``operand2`` is given the ``operand`` will be used as ``self`` for
        the arithmetic operation and ``operand2`` as ``operand``.
        """
        if operand2 is None:
            operand2 = operand
            operand = self

        # Wrap the operands in NDData objects to allow arithmetic operations
        # with numbers, lists, numpy arrays, numpy masked arrays, astropy
        # Quantities or astropy Masked.
        if not isinstance(operand, NDArithmeticMixin):
            if isinstance(operand, Masked):
                operand = operand.unmasked
            operand = self.__class__(operand)
        if not isinstance(operand2, NDArithmeticMixin):
            if isinstance(operand2, Masked):
                operand2 = operand2.unmasked
            operand2 = self.__class__(operand2)

        # Call the arithmetic method
        result, init_kwds = operand._arithmetic(operation, operand2, **kwargs)
        # Return a new class
        return self.__class__(result, **init_kwds)
