# Licensed under a 3-clause BSD style license - see PYFITS.rst

import sys
import warnings
from collections import OrderedDict
from collections.abc import Mapping

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
    _wrapx,
)
from astropy.io.fits.util import _is_int, _is_pseudo_integer, _pseudo_zero
from astropy.utils import isiterable


def _get_recarray_field(array, key):
    """
    Compatibility function for getting a field from a record array.
    
    This handles the deprecation of multi-field indexing in NumPy 1.16+
    while maintaining backward compatibility.
    """
    if isinstance(key, tuple) and len(key) > 1:
        # Multi-field access - handle each field separately
        # and return a new record array with only those fields
        field_names = []
        field_arrays = []
        
        for field_name in key:
            if isinstance(field_name, int):
                # Convert integer index to field name
                field_name = array.dtype.names[field_name]
            field_names.append(field_name)
            field_arrays.append(array[field_name])
        
        # Create new dtype with selected fields
        new_dtype = [(name, array.dtype.fields[name][0]) for name in field_names]
        
        # Create new structured array
        result = np.empty(array.shape, dtype=new_dtype)
        for i, name in enumerate(field_names):
            result[name] = field_arrays[i]
        
        return result
    else:
        # Single field access - use normal indexing
        return array[key]


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
        input : `FITS_rec`
            The array; normally a `FITS_rec` object.

        row : int, optional
            The starting logical row of the array.

        start : int, optional
            The starting column in the row associated with this object.
            Used for subsetting the columns of the `FITS_rec` object.

        end : int, optional
            The ending column in the row associated with this object.
            Used for subsetting the columns of the `FITS_rec` object.
        """
        self.array = input
        self.row = row
        if base:
            width = len(base)
        else:
            width = self.array._nfields

        # This usage of ``range`` is safe for large lists
        s = slice(start, end, step).indices(width)
        self.start, self.end, self.step = s
        self.base = base

    def __getitem__(self, key):
        if isinstance(key, str):
            indx = _get_index(self.array.names, key)

            if indx < self.start or indx > self.end - 1:
                raise KeyError(f"Key {key!r} does not exist.")
        elif isinstance(key, slice):
            return type(self)(
                self.array, self.row, key.start, key.stop, key.step, self
            )
        else:
            indx = self.start + key
            if indx > self.end - 1:
                raise IndexError("Index out of bounds")

        return self.array.field(indx)[self.row]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            indx = _get_index(self.array.names, key)

            if indx < self.start or indx > self.end - 1:
                raise KeyError(f"Key {key!r} does not exist.")
        elif isinstance(key, slice):
            for indx in range(*key.indices(len(self))):
                indx = self.start + indx
                self.array.field(indx)[self.row] = value
            return
        else:
            indx = self.start + key
            if indx > self.end - 1:
                raise IndexError("Index out of bounds")

        self.array.field(indx)[self.row] = value

    def __getslice__(self, start, end):
        return self[slice(start, end)]

    def __len__(self):
        return len(range(self.start, self.end, self.step))

    def __repr__(self):
        """
        Display a single row.
        """
        outlist = []
        for idx in range(len(self)):
            outlist.append(repr(self[idx]))
        return "(" + ", ".join(outlist) + ")"

    def field(self, field):
        """
        Get the field data of the record.
        """
        return self.array.field(field)[self.row]

    def setfield(self, field, value):
        """
        Set the field data of the record.
        """
        self.array.field(field)[self.row] = value


class FITS_rec(np.recarray):
    """
    FITS record array class.

    `FITS_rec` is the data object used by the `~astropy.io.fits.BinTableHDU`
    class.  This inherits from `~numpy.recarray`, but adds some
    functionality to allow us to deal with scaled columns.  It also handles
    conversion/scaling of columns.

    The `FITS_rec` class expects a `FITS_rec` object as input.
    """

    _cache = True

    def __new__(
        subtype,
        input,
        dtype=None,
        shape=None,
        offset=0,
        strides=None,
        formats=None,
        names=None,
        titles=None,
        byteorder=None,
        aligned=False,
        copy=True,
    ):
        """
        Construct a FITS record array from a wide variety of objects.
        """
        if dtype is not None:
            descr = np.dtype(dtype)
        else:
            descr = None

        if isinstance(input, (type(None), str, list, tuple)):
            if isinstance(input, str):
                raise ValueError(
                    "Cannot create a FITS_rec from a string; "
                    "use FITS_rec.from_string() instead"
                )

            # Construct from the dtype
            if descr is None:
                if formats is None:
                    formats = ["E"] * len(names)
                if names is None:
                    names = ["c%d" % n for n in range(len(formats))]

                # Allow for either the formats or the names to be longer, but
                # not both
                if len(formats) < len(names):
                    formats.extend(["E"] * (len(names) - len(formats)))
                elif len(names) < len(formats):
                    names.extend(["c%d" % n for n in range(len(names), len(formats))])

                descr = np.dtype(
                    [(n, _convert_format(f)) for n, f in zip(names, formats)]
                )

            _input = input
            if shape is None:
                shape = (0,)
            input = np.zeros(shape, dtype=descr)
            if _input:
                input = np.rec.fromrecords(_input, dtype=descr)

        # Check the type of the input
        if isinstance(input, FITS_rec):
            # Get the record array
            input = input.view(np.recarray)
        elif isinstance(input, np.ndarray):
            if input.dtype.names is None:
                raise ValueError("Array must have named fields.")
        else:
            input = np.rec.fromrecords(input, dtype=descr)

        # Create the new object
        self = input.view(subtype)

        if copy:
            self = self.copy()

        # Set defaults
        self._init()

        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._init()

    def __reduce__(self):
        """
        Return information needed to pickle the record array.
        """
        return _reconstruct_fits_rec, (np.asarray(self), self.dtype), self.__dict__

    def __setstate__(self, state):
        """
        Restore the internal state of the record array for pickling
        purposes.  This is an internal numpy method.
        """
        self.__dict__.update(state)

    def _init(self):
        """
        Runs after the underlying ndarray is created, subclasses should
        override this rather than __new__ or __array_finalize__ unless they
        need to change the underlying ndarray.
        """
        self._nfields = len(self.dtype.names) if self.dtype.names else 0
        self._convert = [None] * self._nfields
        self._heapoffset = getattr(self, "_heapoffset", 0)
        self._heapsize = getattr(self, "_heapsize", 0)
        self._file = getattr(self, "_file", None)
        self._buffer = None
        self._coldefs = None
        self._gap = 0
        self._uint = False
        self.parnames = None
        self._scale_back = None
        # Column-wise null value masks; if the string representation of a
        # column's null is different from its binary representation, this
        # attribute will be set.
        self._masks = {}
        # Store a copy of the original header, and parse it for scale
        # keywords and alternative column names
        self._header = None
        self._coldefs = None
        self._character_as_bytes = False

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.field(key)
        elif isinstance(key, (slice, np.ndarray, tuple, list)):
            # Have to view as a recarray then back to a FITS_rec, otherwise
            # columns do not convert properly
            if isinstance(key, tuple):
                # Handle multi-field indexing
                return _get_recarray_field(self.view(np.recarray), key).view(type(self))
            else:
                return self.view(np.recarray)[key].view(type(self))
        else:
            return FITS_record(self, key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.field(key)[:] = value
        elif isinstance(key, slice):
            end_idx = min(len(self), key.stop or len(self))
            start_idx = max(0, key.start or 0)
            for idx in range(start_idx, end_idx, key.step or 1):
                self._set_row(idx, value)
        elif isiterable(key):
            for idx in key:
                self._set_row(idx, value)
        else:
            self._set_row(key, value)

    def _set_row(self, idx, value):
        if isinstance(value, FITS_record):
            for field_idx in range(len(value)):
                self.field(field_idx)[idx] = value[field_idx]
        elif isinstance(value, (tuple, list, np.void)):
            if isinstance(value, np.void):
                # This handles the case where we're assigning from a record
                # array element
                for field_idx, field_name in enumerate(self.dtype.names):
                    self.field(field_idx)[idx] = value[field_name]
            else:
                for field_idx, val in enumerate(value):
                    self.field(field_idx)[idx] = val
        else:
            raise TypeError(
                "Cannot set a table row to a scalar value {!r}.".format(value)
            )

    def copy(self, order="C"):
        """
        Return a copy of the record array.
        """
        # This should work for most cases, but we may need to
        # override this method in the future for more complicated
        # cases.
        copied = self.view(np.recarray).copy(order)
        copied = copied.view(type(self))
        copied.__dict__.update(self.__dict__)
        return copied

    def __repr__(self):
        # The addition of the _coldefs to the kwds is a little strange, but
        # it's necessary because it may not be possible to find the
        # information needed to create the column definitions from the FITS
        # header stored in the HDU at the time this method is called.
        kwds = ["dtype={!r}".format(self.dtype)]
        if hasattr(self, "_coldefs") and self._coldefs is not None:
            kwds.append("coldefs={!r}".format(self._coldefs))
        kwds = ", ".join(kwds)
        return f"{self.__class__.__name__}({kwds})\n{self.view(np.recarray)!r}"

    def field(self, key):
        """
        A view of a `Column`'s data as an array.
        """
        # NOTE: The *column* index may not be the same as the *field* index
        # if the column is a phantom column
        column = self.columns[key]
        name = column.name
        format = column.format
        
        # If field does not exist then create a phantom field
        if name not in self.dtype.names:
            if format.dtype == np.dtype("a1"):
                # Char column
                arr = chararray.array(np.empty(self.shape + format.shape,
                                               dtype=format.dtype),
                                      copy=False)
                if len(format.shape) > 0:
                    arr = arr[..., 0]
            else:
                arr = np.empty(self.shape + format.shape, dtype=format.dtype)
                
            # Make the array C-contiguous
            arr = np.ascontiguousarray(arr)
            
            # Store it so that it can be retrieved again without
            # re-creating it
            if not hasattr(self, '_phantom_fields'):
                self._phantom_fields = {}
            self._phantom_fields[name] = arr
            return arr
        
        # TODO: If the field has a different shape than that given by the
        # dtype, we may want to allow that; right now it would cause a
        # ValueError when setting the array scalar below
        base = self.view(np.recarray)[name]
        
        # base is a recarray field, for string columns we may have to
        # convert white spaces to nulls
        if isinstance(format, _AsciiColDefs):
            # TODO: Maybe this should not be using private attributes
            null = format.null
            if null is not None:
                mask = (base == null.encode('ascii'))
                # Return a MaskedArray if nulls are found
                if np.any(mask):
                    base = np.ma.array(base, mask=mask, copy=False)
        
        # Set up the conversion function if not already done
        indx = self.names.index(name)
        if self._convert[indx] is None:
            format = column.format
            # check for scale parameters
            option = (format.scale, format.zero, format.bscale, format.bzero)
            if option != (None, None, None, None):
                self._convert[indx] = self._get_scale_factors(indx)[2]
        
        # Apply conversion on base
        if self._convert[indx]:
            # Make a copy of the field so that we don't modify the original
            # recarray
            if not base.flags.owndata:
                # Ensure we own our own data before modifying
                base = base.copy()
            
            # Check if the conversion function is a simple scale/offset
            # TODO: Numpy 1.6 and up allows this more efficient approach
            # to be used for the most common case
            if hasattr(self._convert[indx], '__call__'):
                # It's a function, call it
                base = self._convert[indx](base)
            else:
                # It should be a tuple of (scale, zero)
                scale, zero = self._convert[indx]
                if scale != 1:
                    np.multiply(base, scale, base)
                if zero != 0:
                    np.add(base, zero, base)
        
        return base

    @property
    def columns(self):
        """
        A user-visible accessor for the coldefs.
        """
        return self._coldefs

    @property
    def _nfields(self):
        """
        Number of fields in the record.
        """
        return len(self.dtype.names) if self.dtype.names else 0

    @_nfields.setter
    def _nfields(self, val):
        # This is a dummy setter to prevent AttributeError when
        # _nfields is set in _init
        pass

    @property
    def names(self):
        if hasattr(self, '_coldefs') and self._coldefs is not None:
            return self._coldefs.names
        else:
            return self.dtype.names

    def _get_scale_factors(self, indx):
        """
        Get the scaling factors for a column.
        
        This method should return a tuple of (scale, zero, func) where
        scale and zero are the scaling parameters and func is a function
        that can be used to convert the column data.
        """
        # This is a simplified version - in practice this would need
        # to handle TSCALn and TZEROn keywords from the header
        return (1.0, 0.0, None)

    @classmethod
    def from_columns(
        cls,
        columns,
        nrows=0,
        fill=False,
        character_as_bytes=False,
        **kwargs
    ):
        """
        Given a `ColDefs` object of unknown type, returns a new `FITS_rec`
        object.

        Parameters
        ----------
        columns : sequence of `Column` or a `ColDefs`
            The columns from which to create the table data.  If these
            columns have data arrays attached that data may be used to
            initialize the new table.  Otherwise the input columns will be
            used as a template for a new table with the requested number of
            rows.

        nrows : int
            Number of rows in the new table.  If the input columns have data
            associated with them, the size of the largest input column is
            used.  Otherwise the default is 0.

        fill : bool
            If `True`, will fill all cells with zeros or blanks.  If
            `False`, copy the data from input, undefined cells will still
            be filled with zeros/blanks.

        character_as_bytes : bool
            Whether to return bytes for string columns when accessed from the
            record array.  By default this is `False` and (unicode) strings
            are returned, but for large tables this may use up a lot of memory.
        """
        from astropy.io.fits.column import ColDefs

        if not isinstance(columns, ColDefs):
            columns = ColDefs(columns)

        # read the delayed data
        for column in columns:
            arr = column.array
            if arr is not None:
                # Make sure to read in the entire array; this is to make sure
                # that it doesn't matter in what order the column arrays are
                # accessed
                dummy = arr.flat[0]
                del dummy

        dtype = columns.dtype
        raw_data = columns._raw_data
        del columns._raw_data

        if raw_data is None:
            if any(col.array is not None for col in columns):
                # Determine the largest array size
                max_len = max(
                    len(col.array) for col in columns if col.array is not None
                )
                nrows = max(nrows, max_len)

            raw_data = np.zeros((nrows,), dtype=dtype)
            if not fill:
                for idx, column in enumerate(columns):
                    # For each column in the ColDef object, determine the
                    # right asciitnull value to fill empty cells with
                    if column.array is not None:
                        raw_data.field(idx)[: len(column.array)] = column.array
                    else:
                        column._blank_fill(raw_data.field(idx))
        else:
            raw_data = raw_data.view(dtype).squeeze()

        hdu = FITS_rec(raw_data, character_as_bytes=character_as_bytes, **kwargs)
        hdu._coldefs = columns
        columns._parent = hdu
        hdu._character_as_bytes = character_as_bytes
        return hdu


def _reconstruct_fits_rec(buf, dtype):
    """
    Reconstruct a FITS_rec from pickled data.
    """
    return buf.view(dtype=dtype, type=FITS_rec)


def _convert_format(format, strict_dimensionality=False):
    """
    Convert FITS format specifications to numpy format specifications.
    """
    # This is a simplified version of format conversion
    # In practice this would need to handle all FITS format codes
    if format in FITS2NUMPY:
        return FITS2NUMPY[format]
    else:
        # Default to float64 for unknown formats
        return 'f8'


# Import chararray here to avoid circular imports
try:
    from numpy import chararray
except ImportError:
    # Fallback for older numpy versions
    chararray = None