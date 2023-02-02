import xarray as xr 
import numpy as np
import itertools
import xtile.rotate as rot 

# Check that shapes are 2D and match
def _check_shapes(sh1, sh2):
    if len(sh1) != 2 or len(sh2) != 2:
        raise ValueError('Layout must be 2D but inputs are {}D and {}D'.format(
            len(sh1), len(sh2)))
    if sh1 != sh2:
        raise ValueError('Inputs have different shapes {} and {}'.format(sh1, sh2))
    return sh1

# Check "compatibility" with a reference dataset. Requires
# - same dimensions (names and size)
# - same coordiantes (names and dimensions)
# - same data variables (names and dimensions)
def _check_compat(dsets, reference_dataset):

    ref = dsets[reference_dataset]

    for dset in dsets.ravel():

        if set(dset.dims.keys()) != set(ref.dims.keys()):
            raise ValueError('Attempt to merge datasets with different dimensionalities.')
        for key in dset.dims.keys():
            if dset.sizes[key] != ref.sizes[key]:
                raise ValueError('Attempt to merge datasets with different shapes.')
        
        if set(dset.coords.keys()) != set(ref.coords.keys()):
            raise ValueError('Attempt to merge datasets with different coordinates.')
        for key in dset.coords.keys():
            if dset.coords[key].dims != ref.coords[key].dims:
                raise ValueError('Attempt to merge datasets with different dimensions on the same coordinate.')
        
        if set(dset.data_vars.keys()) != set(ref.data_vars.keys()):
            raise ValueError('Attempt to merge datasets with different variables.')
        for key in dset.data_vars.keys():
            if dset.data_vars[key].dims != ref.data_vars[key].dims:
                raise ValueError('Attempt to merge datasets with different dimensions on the same variable.')

    return ref

# Check that post-merge coordinates arrays are provided for all concatenation dimensions,
# and compute post-merge coordinate dictionary.
def _check_merge_coords(dset, x, y, global_coords):
    rot._validate_axes(x, y)
    for key in x + y:
        if not key in global_coords:
            raise ValueError('Coordinate {} missing from global_coords'.format(key))
    coords = {}
    for key in dset.coords:
        if key in global_coords:
            coords[key] = xr.DataArray(global_coords[key], coords={key: global_coords[key]},
                dims=(key,), attrs=dset.attrs)
        else:
            coords[key] = dset.coords[key]
    return coords


def open_mfdatasets(files, rotations, x, y, **xr_kwargs):
    """
    Open a 2D array of datasets, applying rotations if necessary.

    Parameters
    ----------
    files: (M,N) array of str or list of str
        File paths, passed to `xr.open_mfdataset` to open each dataset in the 2D array.

    rotations: (M,N) array of int 
        Rotations applied to each dataset after opening. Elements are used as `k` parameters 
        in calls to `xtile.rotate.rot90`.

    x, y: str or list of str
        Dimension names defining rotation planes. The duple (x, y) is passed to `xtile.rotate.rot90`
        as the `axes` parameter.

    Keyword arguments
    -----------------
    **xr_kwargs
        Keywork arguments passed to `xr.open_mfdataset`.

    Returns
    -------
    dsets: (M, N) array of xr.Dataset
        Datasets created from lists of files in each element of `files`, with rotations applied as 
        specified by `rotations`, `x`, and `y`.
    """

    ny, nx = _check_shapes(files.shape, rotations.shape)
    dsets = np.empty((ny,nx), dtype=object)
    for i, j in itertools.product(range(ny), range(nx)):
        unrotated_dset = xr.open_mfdataset(files[i,j], **xr_kwargs)
        dsets[i,j] = rot.rot90(unrotated_dset, rotations[i,j], (x, y))
    return dsets

# Return a list of elements present in both arguments
def _matching_dimensions(dims, other_dims):
    return [d for d in dims if d in other_dims]

# Trim redundant entries in arrays defined on outer edges of the domain
def _trim_outer(darrs, dim):
    for i in range(1, len(darrs)):
        darrs[i] = darrs[i].isel({dim: slice(1,None)})

# Change coordinate labels to nan 
def _make_nan_coord(darrs, dim):
    for i in range(len(darrs)):
        darrs[i] = darrs[i].assign_coords({
            dim: np.full(darrs[i][dim].shape, np.nan)})

# Merge implementation for variables with only a single concatenation dimension
# The set_to_nan argument is required, in some cases, to avoid expanding orthogonal 
# concatenation coordinates when this function is called from _merge_2d.
def _merge_1d(dsets, var, dim, new_coord, outer, set_to_nan=None):
    darrs = [dset[var] for dset in dsets]
    if set_to_nan:
        _make_nan_coord(darrs, set_to_nan)
    if outer:
        _trim_outer(darrs, dim)
    darr = xr.concat(darrs, xr.DataArray(data=new_coord, coords={dim: new_coord}, name=dim))
    return darr

# Merge implementation for variables with two concatenation dimensions
def _merge_2d(dsets, var, xdim, ydim, new_xcoord, new_ycoord, x_outer, y_outer):
    darrs = [
        _merge_1d(dsets[i,:], var, xdim, new_xcoord, x_outer, set_to_nan=ydim)
        for i in range(dsets.shape[0])
    ]
    if y_outer:
        _trim_outer(darrs, ydim)
    darr = xr.concat(darrs, xr.DataArray(data=new_ycoord, coords={ydim: new_ycoord}, name=ydim))
    return darr

def merge(dsets, x, y, global_coords, reference_dataset=(0,0), outer=[]):
    """
    Merge a 2D array of xarray Datasets into a single "global" dataset. Most often
    called using the output of `xtile.layout.open_mfdatasets`.

    Parameters
    ----------
    dsets: (M,N) array of xr.Dataset
        2D array of datasets. All datasets in the array must

        * have identical dimensions (same names and same sizes)
        * have compatible coordinates (same names and same dimensions)
        * have compatible data variables (same names and same dimensions)

    x, y: str or list of str 
        Concatenation dimensions. Dimensions in `x` are used for concatenation 
        along axis 0 of `dsets` (the length-M axis), and dimensions in `y` are
        used for concatenation along axis 1 of `dsets` (the length-N axis).

    global_coords: dict (str -> array)
        Mapping from concatenation dimensions (entries in `x` and `y`) to 
        post-concatenation ("global") coordinates.

    Keyword arguments
    -----------------
    reference_dataset: (int, int), default (0, 0)
        Index in `dsets` of the reference dataset used to resolve ambiguities
        when creating the merged dataset. Used to:

        * set dataset and coordinate attributes.
        * select the row in `dsets` used when concatenating variables
          without a y dimension (assumed to be fields that vary in x only)
        * select the column in `dsets` used when concatenating variables
          without an x dimension (assumed to be fields that vary in y only)
        * select the element in `dsets` used to set values for variables that 
          contain neither an x nor a y dimension (assumed to be fields that 
          are independent of x and y).

    outer: list of str, default []
        Names of dimensions that specify fields defined on outer boundaries
        with overlap between adjacent files. Special care is required when
        merging these fields because redundant values on overlapping
        boundaries have to be trimmed before concatenation.
    
    Returns
    -------
    merged: xr.Dataset 
        "Global" merged dataset.
    """

    ref = _check_compat(dsets, reference_dataset)
    coords = _check_merge_coords(ref, x, y, global_coords)
    
    data_vars = {}
    for var in ref.data_vars:
        
        xname = _matching_dimensions(ref[var].dims, x)
        yname = _matching_dimensions(ref[var].dims, y)

        if len(xname) > 1:
            raise ValueError('Variable {} contains multiple x coordinates'.format(var))
        if len(yname) > 1:
            raise ValueError('Variable {} contains multiple y coordinates'.format(var))

        if len(xname) == 0 and len(yname) == 0:
            data_vars[var] = ref[var]
            continue

        if len(xname) == 0:
            ix = reference_dataset[1]
            dim = yname[0]
            data_vars[var] = _merge_1d(dsets[:,ix], var, dim, global_coords[dim], dim in outer)
            continue

        if len(yname) == 0:
            iy = reference_dataset[0]
            dim = xname[0]
            data_vars[var] = _merge_1d(dsets[iy,:], var, dim, global_coords[dim], dim in outer)
            continue

        xdim = xname[0]
        ydim = yname[0]
        data_vars[var] = _merge_2d(dsets, var, xdim, ydim,
            global_coords[xdim], global_coords[ydim], xdim in outer, ydim in outer)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=ref.attrs)

def index_coords(x, y, shape, sizes, outer=[]):
    """
    Create a set of index coordinates suitable for use as the `global_coords` argument 
    of `merge`, assuming that all tiles within a layout have the same shape.

    Parameters
    ----------
    x, y: str or list of str 
        Names of x and y index coordinates 

    shape: (int, int)
        Shape of the layout. Elements are the number of tiles in x and y 
        directions, respectively.

    sizes: dict (str -> int)
        Mapping from coordinate names to sizes within each tile. This function 
        assumes that all tiles have the same shape, so the global coordinate for 
        a coordinate with name `c` will be `range(1, shape[i]*sizes['c'] + 1)`,
        where `i = 0` if `c` is in `y` and `i = 1` if `c` is in `x`.

    Keyword arguments
    -----------------
    outer: str or list of str, default = []
        If provided, any coordinates listed in `outer` will be treated as coordinates 
        defined with overlap on outer edges with tiles, and the global coordinate for
        a coordinate with name `c` will be `range(1, shape[i]*(sizes['c'] - 1) + 2)`.
    
    Returns
    -------
    global_coords: dict (str -> int)
        Mapping from concatenation dimensions (entries in `x` and `y`) to 
        post-concatenation ("global") coordinates.
    """
    if len(shape) != 2:
        raise ValueError('shape must be 2D but is {}D'.format(len(shape)))
    if isinstance(x, str):
        x = (x,)
    if isinstance(y, str):
        y = (y,)
    global_coords = {}
    for c in x:
        if c in outer:
            global_coords[c] = np.arange(1, shape[1]*(sizes[c] - 1) + 2)
        else:
            global_coords[c] = np.arange(1, shape[1]*sizes[c] + 1)
    for c in y:
        if c in outer:
            global_coords[c] = np.arange(1, shape[0]*(sizes[c] - 1) + 2)
        else:
            global_coords[c] = np.arange(1, shape[0]*sizes[c] + 1)
    return global_coords
