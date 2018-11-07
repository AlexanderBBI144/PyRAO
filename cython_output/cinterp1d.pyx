import numpy as np
cimport numpy as np
cimport cython

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t

@cython.boundscheck(False)  #turn of bounds-checking for entire function
@cython.wraparound(False)   #turn of bounds-checking for entire function
cpdef cinterp1d(np.ndarray[DTYPEf_t, ndim=1] x, np.ndarray[DTYPEf_t, ndim=3] y,
               np.ndarray[DTYPEf_t, ndim=1] new_x):
    """
    cinterp1d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 3D array,
    according to new values from a 1D array new_x. Thus, interpolate
    y[i, j, :] for new_x[i, j].

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 3-D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 1-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 3-D ndarray
        Interpolated values.
    """
    cdef int nx = y.shape[0]
    cdef int ny = y.shape[1]
    cdef int nz = y.shape[2]
    cdef int nx_new = new_x.shape[0]
    cdef int i, j, k
    cdef np.ndarray[DTYPEf_t, ndim=3] new_y = np.zeros((nx_new, ny, nz), dtype=DTYPEf)

    for o in range(nx_new):
        for j in range(ny):
            for k in range(nz):
                for i in range(nx-1):
                    if x[i+1] > new_x[o] and x[i] <= new_x[o]:
                        new_y[o,j,k] = y[i,j,k] + (y[i+1,j,k] - y[i,j,k]) / (x[i+1] - x[i]) * (new_x[o] - x[i])
                        break

    return new_y