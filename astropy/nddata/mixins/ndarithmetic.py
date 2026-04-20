# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
import numpy as np

from ...utils.exceptions import AstropyUserWarning
from ...utils.misc import isiterable


__all__ = ['NDArithmeticMixin']


# We use this global variable to control the behavior of arithmetic operations
# when the result is a scalar. If True, we return a scalar; if False, we return
# a 0-d array.
_RETURN_SCALAR = True


class NDArithmeticMixin(object):
    """
    Mixin class to add arithmetic to an NDData object.

    When subclassing, be sure to list the superclasses in the correct order
    so that the subclass sees NDData as the main superclass. See
    `~astropy.nddata.NDDataArray` for an example.

    Notes
    -----
    This class only aims at covering the most common cases so there are certain
    restrictions on the saved attributes::

        - ``uncertainty`` : has to be something that has a `uncertainty_type`
          property that can be used to be interpreted as variance or standard
          deviation.

        - ``mask`` : has to be something that can be used by a bitwise ``or``
          operation.

        - ``wcs`` : has to implement a way of comparing with ``=`` to allow
          the operation.

        - ``meta`` : has to be a `dict`-like object

        - ``unit`` : has to be a `~astropy.units.Unit` like object

    But there is a workaround that allows to disable handling a specific
    attribute and to simply set the results attribute to ``None`` or to copy
    the existing attribute (and neglect the other).
    For example for ``uncertainty`` not representing an variance or standard
    deviation::

        >>> from astropy.nddata import NDDataArray
        >>> ndd = NDDataArray([1,2,3])
        >>> ndd.uncertainty = "neither std nor var"
        >>> ndd.add([1, 1, 1], handle_uncertainty=np.copy)
        NDDataArray([2, 3, 4])

    """

    def _arithmetic_mask(self, operation, operand, handle_mask, axis=None, **kwds):
        """
        Calculate the resulting mask.

        This is implemented as the piecewise ``or`` of the masks of the
        operands, but if a different behavior is desired, then this method
        can be overridden in a subclass.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin._arithmetic` parameter description.
            By default, the ``operation`` will be ignored.

        operand : same type as self
            see `NDArithmeticMixin._arithmetic` parameter description.

        handle_mask : callable
            see `NDArithmeticMixin._arithmetic` parameter description.

        kwds :
            Additional parameters given to ``handle_mask``.

        Returns
        -------
        result_mask : any type
            If ``None`` then the result will have no mask. Otherwise the
            shape must match the shape of the result.

        Raises
        ------
        ValueError
            If the ``mask`` cannot be broadcast to the shape of the data.
        """
        # If no mask is present in either operand, return None
        if self.mask is None and getattr(operand, 'mask', None) is None:
            return None
        
        # If only one operand has a mask, return that mask
        if self.mask is None:
            return getattr(operand, 'mask', None)
        if getattr(operand, 'mask', None) is None:
            return self.mask
        
        # Both operands have masks, use the handle_mask function
        return handle_mask(self.mask, operand.mask, **kwds)

    def _arithmetic_uncertainty(self, operation, operand, handle_uncertainty, axis=None, **kwds):
        """
        Calculate the resulting uncertainty.

        This implementation uses the ``uncertainty_type`` property to
        determine the correct propagation of uncertainties.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin._arithmetic` parameter description.

        operand : same type as self
            see `NDArithmeticMixin._arithmetic` parameter description.

        handle_uncertainty : callable
            see `NDArithmeticMixin._arithmetic` parameter description.

        kwds :
            Additional parameters given to ``handle_uncertainty``.

        Returns
        -------
        result_uncertainty : `~astropy.nddata.NDUncertainty` subclass instance or None
            The resulting uncertainty, if any.

        Raises
        ------
        ValueError
            If the ``uncertainty`` cannot be broadcast to the shape of the
            data.
        """
        # If no uncertainty is present in either operand, return None
        if self.uncertainty is None and getattr(operand, 'uncertainty', None) is None:
            return None
        
        # If only one operand has uncertainty, return that uncertainty
        if self.uncertainty is None:
            return getattr(operand, 'uncertainty', None)
        if getattr(operand, 'uncertainty', None) is None:
            return self.uncertainty
        
        # Both operands have uncertainties, use the handle_uncertainty function
        return handle_uncertainty(self.uncertainty, operand.uncertainty, **kwds)

    def _arithmetic_wcs(self, operation, operand, handle_wcs, axis=None, **kwds):
        """
        Calculate the resulting wcs.

        There is actually no calculation involved but it is a good place to
        add a warning if the wcs attributes are not identical.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin._arithmetic` parameter description.
            By default, the ``operation`` will be ignored.

        operand : same type as self
            see `NDArithmeticMixin._arithmetic` parameter description.

        handle_wcs : callable
            see `NDArithmeticMixin._arithmetic` parameter description.

        kwds :
            Additional parameters given to ``handle_wcs``.

        Returns
        -------
        result_wcs : any type
            The resulting wcs.
        """
        # ok, not really arithmetic but we need to check which wcs makes sense
        # for the result and this is an ideal place to do it.
        if self.wcs is None and getattr(operand, 'wcs', None) is None:
            return None
        elif self.wcs is None:
            return deepcopy(operand.wcs)
        elif getattr(operand, 'wcs', None) is None:
            return deepcopy(self.wcs)
        else:
            if self.wcs == operand.wcs:
                return deepcopy(self.wcs)
            else:
                warnings.warn("WCS are not identical. Using WCS from first "
                               "operand.", AstropyUserWarning)
                return deepcopy(self.wcs)

    def _arithmetic_meta(self, operation, operand, handle_meta, axis=None, **kwds):
        """
        Calculate the resulting meta.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin._arithmetic` parameter description.
            By default, the ``operation`` will be ignored.

        operand : same type as self
            see `NDArithmeticMixin._arithmetic` parameter description.

        handle_meta : callable
            see `NDArithmeticMixin._arithmetic` parameter description.

        kwds :
            Additional parameters given to ``handle_meta``.

        Returns
        -------
        result_meta : any type
            The resulting meta.
        """
        # Just return what handle_meta does with both of the metas.
        return handle_meta(self.meta, getattr(operand, 'meta', None), **kwds)

    def _arithmetic(self, operation, operand, propagate_uncertainties=True,
                    handle_mask=np.logical_or, handle_meta=None, **kwds):
        """
        Base method which calculates the result of the arithmetic operation.

        This method determines the result of the arithmetic operation on the
        ``data`` including their units and then forwards the call to other
        methods to calculate the other attributes.

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

        handle_mask : callable, ``None`` or ``False``, optional
            see :meth:`NDArithmeticMixin.add`

        handle_meta : callable, ``None`` or ``False``, optional
            see :meth:`NDArithmeticMixin.add`

        kwds :
            Additional parameters.

        Returns
        -------
        result : same type (class) as self
            The resulting dataset

        Notes
        -----
        The unit handling follows the rules for `~astropy.units.Quantity` if
        units are present on both operands.
        """
        # Find the appropriate keywords for the appropriate method
        kwds2 = {"handle_mask": handle_mask,
                 "handle_meta": handle_meta}
        kwds2.update(kwds)

        # First check that the WCS allows the arithmetic operation
        if self.wcs is not None or getattr(operand, 'wcs', None) is not None:
            self._arithmetic_wcs(operation, operand, None)

        # Then calculate the resulting data (which can but doesn't need to be a
        # quantity)
        result = operation(self.data, operand)

        # preserve original units
        if hasattr(self, 'unit') and hasattr(operand, 'unit'):
            if self.unit is None:
                kwargs['unit'] = operand.unit
            elif operand.unit is None:
                kwargs['unit'] = self.unit
            else:
                kwargs['unit'] = self.unit
        elif hasattr(self, 'unit'):
            kwargs['unit'] = self.unit
        elif hasattr(operand, 'unit'):
            kwargs['unit'] = operand.unit
        else:
            pass

        # Determine the other attributes
        if propagate_uncertainties is None:
            kwargs['uncertainty'] = None
        elif not propagate_uncertainties:
            if self.uncertainty is None:
                kwargs['uncertainty'] = deepcopy(getattr(operand,
                                                          'uncertainty', None))
            else:
                kwargs['uncertainty'] = deepcopy(self.uncertainty)
        else:
            kwargs['uncertainty'] = self._arithmetic_uncertainty(
                operation, operand, propagate_uncertainties, **kwds2)

        if handle_mask is None:
            kwargs['mask'] = None
        elif handle_mask is False:
            if self.mask is None:
                kwargs['mask'] = deepcopy(getattr(operand, 'mask', None))
            else:
                kwargs['mask'] = deepcopy(self.mask)
        else:
            kwargs['mask'] = self._arithmetic_mask(operation, operand,
                                                   handle_mask, **kwds2)

        if handle_meta is None:
            kwargs['meta'] = None
        elif handle_meta is False:
            if self.meta is None:
                kwargs['meta'] = deepcopy(getattr(operand, 'meta', None))
            else:
                kwargs['meta'] = deepcopy(self.meta)
        else:
            kwargs['meta'] = self._arithmetic_meta(operation, operand,
                                                   handle_meta, **kwds2)

        # Wrap the individual results into the same class
        return self.__class__(result, **kwargs)

    def add(self, operand, operand2=None, **kwargs):
        """
        Performs addition by evaluating ``self`` + ``operand``.

        Parameters
        ----------
        operand, operand2 : `NDData`-like instance
            If ``operand2`` is ``None`` or not given, the operation is simply
            ``self`` + ``operand``.
            If ``operand2`` is given, the operation is ``self`` + ``operand`` +
            ``operand2``.
            If the method should have additional behavior for two operands it
            is recommended to call the method with ``operand2``
            explicitly even if only one operand is given.

        propagate_uncertainties : `bool` or ``None``, optional
            If ``None`` the result will have no uncertainty. If ``False`` the
            result will have a copied version of the first operand that has an
            uncertainty. If ``True`` the result will have a correctly
            propagated uncertainty from the uncertainties of the operands but
            this assumes that the uncertainties are `NDUncertainty`-like. If
            the operand's uncertainties are not `NDUncertainty`-like or if
            arithmetic propagation of uncertainties is not possible/wanted one
            can use :meth:`NDArithmeticMixin._arithmetic_uncertainty` to
            calculate the uncertainties manually.
            Default is ``True``.

            .. versionchanged:: 1.2
                This parameter must be given as keyword-parameter. Using it as
                positional parameter is deprecated.
                ``None`` was added as valid parameter value.

        handle_mask : callable, ``None`` or ``False``, optional
            If ``None`` the result will have no mask. If ``False`` the
            result will have a copied version of the first operand that has a
            mask). If it is a callable then the specified callable must
            create the results ``mask`` and if necessary provide a ``where``
            parameter if the result is a `numpy.ma.MaskedArray`.
            Default is `numpy.logical_or`.

            .. versionadded:: 1.2

        handle_meta : callable, ``None`` or ``False``, optional
            If ``None`` the result will have no meta. If ``False`` the
            result will have a copied version of the first operand that has a
            (not empty) meta. If it is a callable then the specified callable
            must create the results ``meta``.
            Default is ``None``.

            .. versionadded:: 1.2

        kwargs :
            Any other parameter that should be passed to the callables used.

        Returns
        -------
        result : `~astropy.nddata.NDData`-like
            The resulting dataset

        Notes
        -----
        If a ``callable`` is used for ``mask``, ``wcs`` or ``meta`` the
        callable must accept the corresponding attributes as first two
        parameters. If the callable also needs additional parameters these can
        be defined as ``kwargs`` and must start with the name of the callable
        followed by an underscore (``"_"``) followed by the parameter name.
        For example to pass the parameter ``q`` to ``handle_mask`` use the
        parameter name ``handle_mask_q``.
        """
        return self._arithmetic(np.add, operand, **kwargs)

    def subtract(self, operand, **kwargs):
        """
        Performs subtraction by evaluating ``self`` - ``operand``.

        Parameters
        ----------
        operand : `NDData`-like instance
            The second operand in the subtraction.

        propagate_uncertainties : `bool` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`

        handle_mask : callable, ``None`` or ``False``, optional
            see :meth:`NDArithmeticMixin.add`

        handle_meta : callable, ``None`` or ``False``, optional
            see :meth:`NDArithmeticMixin.add`

        kwargs :
            Any other parameter that should be passed to the callables used.

        Returns
        -------
        result : `~astropy.nddata.NDData`-like
            The resulting dataset
        """
        return self._arithmetic(np.subtract, operand, **kwargs)

    def multiply(self, operand, **kwargs):
        """
        Performs multiplication by evaluating ``self`` * ``operand``.

        Parameters
        ----------
        operand : `NDData`-like instance
            The second operand in the multiplication.

        propagate_uncertainties : `bool` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`

        handle_mask : callable, ``None`` or ``False``, optional
            see :meth:`NDArithmeticMixin.add`

        handle_meta : callable, ``None`` or ``False``, optional
            see :meth:`NDArithmeticMixin.add`

        kwargs :
            Any other parameter that should be passed to the callables used.

        Returns
        -------
        result : `~astropy.nddata.NDData`-like
            The resulting dataset
        """
        return self._arithmetic(np.multiply, operand, **kwargs)

    def divide(self, operand, **kwargs):
        """
        Performs division by evaluating ``self`` / ``operand``.

        Parameters
        ----------
        operand : `NDData`-like instance
            The second operand in the division.

        propagate_uncertainties : `bool` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`

        handle_mask : callable, ``None`` or ``False``, optional
            see :meth:`NDArithmeticMixin.add`

        handle_meta : callable, ``None`` or ``False``, optional
            see :meth:`NDArithmeticMixin.add`

        kwargs :
            Any other parameter that should be passed to the callables used.

        Returns
        -------
        result : `~astropy.nddata.NDData`-like
            The resulting dataset
        """
        return self._arithmetic(np.true_divide, operand, **kwargs)

    def __add__(self, operand):
        return self.add(operand)

    def __radd__(self, operand):
        return self.add(operand)

    def __sub__(self, operand):
        return self.subtract(operand)

    def __rsub__(self, operand):
        return self.__class__(operand).subtract(self)

    def __mul__(self, operand):
        return self.multiply(operand)

    def __rmul__(self, operand):
        return self.multiply(operand)

    def __truediv__(self, operand):
        return self.divide(operand)

    def __rtruediv__(self, operand):
        return self.__class__(operand).divide(self)

    def __div__(self, operand):
        return self.divide(operand)

    def __rdiv__(self, operand):
        return self.__class__(operand).divide(self)