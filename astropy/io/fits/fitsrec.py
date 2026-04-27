# Licensed under a 3-clause BSD style license - see PYFITS.rst

import sys
import warnings
import re
from collections import OrderedDict

import numpy as np

from astropy.io.fits.column import (
    ASCIITNULL,
    FITS2NUMPY,
    NUMPY2FITS,
    _AsciiColDefs,
    _FormatP,
    _FormatQ,
    _VLF,
    _get_index,
    _makep,
    _parse_tdisp_format,
    _scalar_to_format_string,
    _VarLenArrayDef,
)
from astropy.io.fits.util import (
    _is_pseudo_integer,
    _pseudo_zero,
    _str_to_num,
    decode_ascii,
    encode_ascii,
)
from astropy.utils import lazyproperty


__all__ = ["FITS_rec", "FITS_record"]


# mapping from FITS format to numpy format
_FORMATRE = re.compile(r"(?P<repeat>\d+)?(?P<format>[LXBIJKAEDCMPQ]?)(?P<option>.*)?")


class FITS_record:
    """
    FITS record class.

    `FITS_record` is used to access records of the `FITS_rec` object.
    This will allow us to deal with scaled columns.  It also handles
    conversion/scaling of columns.

    The `FITS_record` class expects a `FITS_rec` object as input.
    """

    def __init__(self, input, row=0, start=None, end=None, step=None, base=None):
        """
        Parameters
        ----------
        input : array
           The array; should be a `FITS_rec` object.

        row : int, optional
           Row index of the desired record.  Defaults to 0.

        start : int, optional
           Record slice start index.  Defaults to `None`.

        end : int, optional
           Record slice end index.  Defaults to `None`.

        step : int, optional
           Record slice step index.  Defaults to `None`.

        base : `FITS_record`, optional
           `FITS_record` object from which the current record is derived.
           Defaults to `None`.
        """
        self.array = input
        self.row = row
        if base is None:
            self._base = self
        else:
            self._base = base
        self._start = start
        self._end = end
        self._step = step

    def __getitem__(self, key):
        if isinstance(key, str):
            indx = _get_index(self.array.names, key)

            if indx < self.array._nfields:
                column = self.array.columns[indx]

                # If the column is a VLA, we want to return the data for
                # the specific row
                if isinstance(column, _VarLenArrayDef):
                    return column[self.row]
                else:
                    return self.array.field(indx)[self.row]
        else:
            return self.array[self.row][key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            indx = _get_index(self.array.names, key)
            self.array[self.row][indx] = value
        else:
            self.array[self.row][key] = value

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        try:
            self[attr] = value
        except (KeyError, IndexError):
            object.__setattr__(self, attr, value)

    def __repr__(self):
        # Copied from numpy.void.__repr__, but with our own
        # object class name
        return repr(tuple(self)).replace("(", "FITS_record(", 1)

    def __str__(self):
        # Copied from numpy.void.__str__, but with our own
        # object class name
        return str(tuple(self)).replace("(", "FITS_record(", 1)

    def __len__(self):
        return len(self.array._coldefs)

    def __eq__(self, other):
        if isinstance(other, FITS_record):
            return tuple(self) == tuple(other)
        else:
            return tuple(self) == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(self))

    def __iter__(self):
        for indx in range(len(self)):
            yield self[indx]

    def __getslice__(self, start, end):
        return self[start:end]

    def field(self, key):
        """
        Get the field data of the record.
        """
        return self.__getitem__(key)

    def setfield(self, key, value):
        """
        Set the field data of the record.
        """
        self.__setitem__(key, value)


class FITS_rec(np.recarray):
    """
    FITS record array class.

    `FITS_rec` is the data part of a table HDU's data part.  This is a layer
    over the `~numpy.recarray`, so we can deal with scaled columns.

    It inherits all of the standard methods from `numpy.ndarray`.
    """

    _record_type = FITS_record
    _coldefs = None
    _character_as_bytes = False
    _bzero = None
    _bscale = None
    _uint = False
    _heapoffset = 0
    _heapsize = 0
    _file = None
    _buffer = None
    _gap = 0
    _pseudo_uint8_array = None

    def __new__(subtype, input):
        """
        Construct a FITS record array from a recarray.
        """
        # input should be a record array
        if input is not None:
            input = np.asarray(input)
            if input.dtype.names is None:
                raise TypeError("FITS_rec input array must be a record array.")
        else:
            # a record array is still created even if the input is None
            # this makes it easier to create an empty record array without
            # having to specify a dtype
            input = np.array(None, dtype=np.dtype([]))

        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        self = np.asarray(input).view(subtype)

        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return

        if isinstance(obj, FITS_rec):
            self._coldefs = obj._coldefs
            self._character_as_bytes = obj._character_as_bytes
            self._bzero = obj._bzero
            self._bscale = obj._bscale
            self._uint = obj._uint
            self._heapoffset = obj._heapoffset
            self._heapsize = obj._heapsize
            self._file = obj._file
            self._buffer = obj._buffer
            self._gap = obj._gap
            self._pseudo_uint8_array = obj._pseudo_uint8_array

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.field(key)
        elif isinstance(key, int):
            # Have to view as a FITS_record
            out = self.view(np.recarray)[key].view(self._record_type)
            out.array = self
            out.row = key
            return out
        elif isinstance(key, slice) or isinstance(key, np.ndarray):
            # Have to return a FITS_rec
            out = self.view(np.recarray)[key].view(type(self))
            out._coldefs = self._coldefs
            out._character_as_bytes = self._character_as_bytes
            arrays = []
            out._arrays = arrays
            if self._coldefs is not None:
                for idx, col in enumerate(self._coldefs):
                    # Store a reference to the new array object in the
                    # column itself
                    arrays.append(out.view(np.recarray).field(idx))
                    col.array = arrays[idx]
            return out
        else:
            return super().__getitem__(key)

    def __getslice__(self, start, end):
        return self[start:end]

    def copy(self, order="C"):
        """
        Return a copy of the array.
        """
        copied = self.view(np.recarray).copy(order)
        return copied.view(self.__class__)

    def __reduce__(self):
        """
        Return the pickle state.

        Restores by passing the record array data to the constructor.
        """
        return (_fits_rec_from_array, (self.view(np.recarray),))

    @classmethod
    def from_columns(
        cls,
        columns,
        nrows=0,
        fill=False,
        character_as_bytes=True,
        **kwargs
    ):
        """
        Given either a `ColDefs` object, a sequence of `Column` objects, or
        another `FITS_rec` object, return a new `FITS_rec` object.

        See `new_table` for a complete description of the parameters.
        """
        from astropy.io.fits.hdu.table import new_table

        if isinstance(columns, FITS_rec):
            # Just copy the original
            return columns.copy()

        # This is to support the deprecated API
        if "tbtype" in kwargs:
            if kwargs["tbtype"] == "TableHDU":
                character_as_bytes = True
            elif kwargs["tbtype"] == "BinTableHDU":
                character_as_bytes = False
            del kwargs["tbtype"]

        hdu = new_table(
            columns, nrows=nrows, fill=fill, character_as_bytes=character_as_bytes
        )
        return hdu.data

    @lazyproperty
    def columns(self):
        """
        A user-visible accessor for the coldefs.
        """
        return self._coldefs

    @lazyproperty
    def _nfields(self):
        """
        Number of fields in the record.
        """
        return len(self.dtype.names) if self.dtype.names else 0

    @lazyproperty
    def names(self):
        """
        List of column names.
        """
        if self.dtype.names:
            return list(self.dtype.names)
        return []

    @lazyproperty
    def formats(self):
        """
        List of column formats.
        """
        if self._coldefs is not None:
            return [c.format for c in self._coldefs]
        return []

    def field(self, key):
        """
        A view of a `Column`'s data as an array.
        """
        # NOTE: The behavior of this method is subtly different from the
        # field method of plain recarrays, because for FITS_rec it returns
        # the raw recarray field, whereas this method may return a scaled
        # view of the data.  The naming, however, should be consistent.
        column = self.columns[key]
        name = column.name
        format = column.format

        # If the column is a VLA we want to return the column itself, so that
        # it can deal with returning the correct rows.
        if isinstance(column, _VarLenArrayDef):
            return column

        field = np.recarray.field(self, _get_index(self.names, key))
        if column.null is not None:
            field = _get_masked_view(field, column.null)

        # ASCII table, replace exponent separator 'D' with 'E'
        if isinstance(self._coldefs, _AsciiColDefs):
            if "D" in format.upper():
                # Fix the bug: assign the result back to field
                field = np.char.replace(field, b"D", b"E")
                field = np.char.replace(field, b"d", b"e")

        # Deal with unsigned integer 8-bit, 16-bit, 32-bit, and 64-bit columns
        if column.bscale is not None or column.bzero is not None:
            field = self._get_scaled_view(field, column.bscale, column.bzero)

        # Deal with X format
        if isinstance(format, _FormatX):
            field = self._get_X_view(field)

        # Deal with P format
        elif isinstance(format, _FormatP):
            field = self._get_P_view(field, format)

        # Deal with Q format
        elif isinstance(format, _FormatQ):
            field = self._get_Q_view(field, format)

        return field

    def _get_scaled_view(self, field, bscale, bzero):
        """
        Get a scaled view of the field.
        """
        if bscale != 1:
            field = field * bscale
        if bzero != 0:
            field = field + bzero
        return field

    def _get_X_view(self, field):
        """
        Get the X format view of the field.
        """
        # For X format, return the field as-is
        return field

    def _get_P_view(self, field, format):
        """
        Get the P format view of the field.
        """
        # For P format, return the field as-is for now
        return field

    def _get_Q_view(self, field, format):
        """
        Get the Q format view of the field.
        """
        # For Q format, return the field as-is for now
        return field


def _get_masked_view(field, null_value):
    """
    Get a masked view of the field where null values are masked.
    """
    import numpy.ma as ma
    return ma.array(field, mask=(field == null_value))


def _fits_rec_from_array(array):
    """
    Create a FITS_rec from a numpy record array.
    """
    return array.view(FITS_rec)


class _FormatX:
    """
    Represents the X format for bit arrays.
    """
    pass