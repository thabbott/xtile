import xarray as xr 
import numpy as np 
import dask.array as da 
from enum import Enum

def _validate_axes(x, y):
    if isinstance(x, str):
        x = [x]
    if isinstance(y, str):
        y = [y]
    if len(x) != len(y):
        raise ValueError('Length of x ({}) and y ({}) differ'.format(len(x), len(y)))
    xset = set(x)
    if len(x) != len(xset):
        raise ValueError('x contains duplicate coordinates')
    yset = set(y)
    if len(y) != len(yset):
        raise ValueError('y contains duplicate coordinates')
    common_elements = xset.intersection(yset)
    if len(common_elements) != 0:
        raise ValueError('x and y contain commom coordinates (example: {})'.format(
            common_elements[0]))
    return x, y

def rot90(dset_or_darr, k, axes):
    if isinstance(dset_or_darr, xr.Dataset):
        return _rot90_dset(dset_or_darr, k, axes)
    if isinstance(dset_or_darr, xr.DataArray):
        return _rot90_darr(dset_or_darr, k, axes)
    return TypeError('dset_or_darr is not an xarray Dataset or DataArray')

def _index_of_dimension(names, dims):
    return [dims.index(n) for n in names if n in dims]

def _rot90_dset(dset, k, axes):
    data_vars = {}
    for var in dset.data_vars:
        data_vars[var] = _rot90_darr(dset[var], k, axes, dset.coords)
    return xr.Dataset(data_vars=data_vars, coords=dset.coords, attrs=dset.attrs)

def _rot90_darr(darr, k, axes, coords):

    # Normalize number of rotations to [0,3]
    # Can return original DataArray if no rotations are required
    k = k % 4
    if k == 0:
        return darr

    # Find locations of rotation axes in DataArray dimensions
    # Raise a ValueError if dimensions contain multiple x or y axes,
    # because the rotation axes are ambiguous in this case.
    x, y = _validate_axes(*axes)
    ix_data = _index_of_dimension(x, darr.dims)
    iy_data = _index_of_dimension(y, darr.dims)
    if len(ix_data) > 1:
        raise ValueError('DataArray includes x dimensions at multiple indices ({})'.format(ix))
    if len(iy_data) > 1:
        raise ValueError('DataArray includes y dimensions at multiple indices ({})'.format(iy))

    # If the DataArray contains no rotation axes, can return the original DataArray
    if len(ix_data) == 0 and len(iy_data) == 0:
        return darr
    
    # If the DataArray contains only a y axis ("rotate toward"), may need to...
    # - rename the dimension   (k = 1 or k = 3)
    # - reverse the dimension  (k = 1 or k = 2)
    if len(ix_data) == 0:
        return _rename_reverse(darr, k, y, x, coords, iy_data[0], (1, 3), (1, 2))
    
    # If the DataArray contains only a x axis ("rotate away from"), may need to...
    # - rename the dimension   (k = 1 or k = 3)
    # - reverse the dimension  (k = 2 or k = 3)
    if len(iy_data) == 0:
        return _rename_reverse(darr, k, x, y, coords, ix_data[0], (1, 3), (2, 3))

    # If the DataArray contains both axes, just need to rotate the data without renaming
    return _rotate(darr, k, ix_data[0], iy_data[0])

def _rename_reverse(darr, k, old_ax_names, new_ax_names, dset_coords, iax_data, krename, kreverse):

    # Rename dimensions if needed
    new_dims = list(darr.dims)
    if k in krename:
        iax_axes = old_ax_names.index(darr.dims[iax_data])
        new_dims[iax_data] = new_ax_names[iax_axes]
    new_dims = tuple(new_dims)

    # Reverse along dimensions if needed
    new_data = darr.data
    if k in kreverse:
        if isinstance(new_data, da.Array):
            flip = da.flip
        elif isinstance(new_data, np.ndarray):
            flip = np.flip
        else:
            raise ValueError('DataArray backed by unsupported type {}'.format(type(new_data)))
        new_data = flip(new_data, axis=iax_data)

    new_coords = dict(zip(new_dims, (dset_coords[d] for d in new_dims)))
    return xr.DataArray(data=new_data, dims=new_dims, coords=new_coords, attrs=darr.attrs)

def _rotate(darr, k, iax_from, iax_to):

    new_data = darr.data
    if isinstance(new_data, da.Array):
        rot90 = da.rot90
    elif isinstance(new_data, np.ndarray):
        rot90 = np.rot90
    else:
        raise ValueError('DataArray backed by unsupported type {}'.format(type(new_data)))
    new_data = rot90(new_data, k, axes=(iax_from, iax_to))

    return xr.DataArray(data=new_data, dims=darr.dims, coords=darr.coords, attrs=darr.attrs)
