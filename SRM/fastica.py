"""
Python implementation of the fast ICA algorithms.
Reference: Tables 8.3 and 8.4 page 196 in the book:
Independent Component Analysis, by  Hyvarinen et al.

Derived from sklearn default implementation
obtained on 3/24/2015. Modified for multi-subject "block" ICA.
"""

# Authors: Pierre Lafaye de Micheaux, Stefan van der Walt, Gael Varoquaux,
#          Bertrand Thirion, Alexandre Gramfort, Denis A. Engemann
# License: BSD 3 clause
import warnings
import numpy as np
from scipy import linalg
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.externals.six import moves
from sklearn.utils import as_float_array, check_random_state
fast_dot = np.dot
#from sklearn.utils.extmath import fast_dot
#from sklearn.utils.validation import check_is_fitted

__all__ = ['fastica', 'FastICA']

def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)

def _shape_repr(shape):
    """Return a platform independent representation of an array shape
    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).
    Under Windows 64 bit (and Python 2), the `long` type is used by default
    in numpy shapes even when the integer dimensions are well below 32 bit.
    The platform specific type causes string messages or doctests to change
    from one platform to another which is not desirable.
    Under Python 3, there is no more `long` type so the `L` suffix is never
    introduced in string representation.
    >>> _shape_repr((1, 2))
    '(1, 2)'
    >>> one = 2 ** 64 / 2 ** 64  # force an upcast to `long` under Python 2
    >>> _shape_repr((one, 2 * one))
    '(1, 2)'
    >>> _shape_repr((1,))
    '(1,)'
    >>> _shape_repr(())
    '()'
    """
    if len(shape) == 0:
        return "()"
    joined = ", ".join("%d" % e for e in shape)
    if len(shape) == 1:
        # special notation for singleton tuples
        joined += ','
    return "(%s)" % joined

def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)

def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite):
    """Convert a sparse matrix to a given format.
    Checks the sparse format of spmatrix and converts if necessary.
    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.
    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.
    dtype : string, type or None
        Data type of result. If None, the dtype of the input is preserved.
    copy : boolean
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean
        Whether to raise an error on np.inf and np.nan in X.
    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, six.string_types):
        accept_sparse = [accept_sparse]

    if accept_sparse is False:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' "
                             "as a tuple or list, it must contain at "
                             "least one string value.")
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
            # create new with correct sparse
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError("Parameter 'accept_sparse' should be a string, "
                         "boolean or list of strings. You provided "
                         "'accept_sparse={}'.".format(accept_sparse))

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data)
    return spmatrix

def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is converted to an at least 2D numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.
    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.
    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.
    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2d.
    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.
    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.
    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.
    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    # accept_sparse 'None' deprecation check
    if accept_sparse is None:
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        accept_sparse = False

    # store whether originally we wanted numeric dtype
    dtype_numeric = dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy,
                                      force_all_finite)
    else:
        array = np.array(array, dtype=dtype, order=order, copy=copy)

        if ensure_2d:
            if array.ndim == 1:
                raise ValueError(
                    "Got X with X.ndim=1. Reshape your data either using "
                    "X.reshape(-1, 1) if your data has a single feature or "
                    "X.reshape(1, -1) if it contains a single sample.")
            array = np.atleast_2d(array)
            # To ensure that array flags are maintained
            array = np.array(array, dtype=dtype, order=order, copy=copy)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array)

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array


def _gs_decorrelation(w, W, j):
    """
    Orthonormalize w wrt the first j rows of W
    Parameters
    ----------
    w : ndarray of shape(n)
        Array to be orthogonalized
    W : ndarray of shape(p, n)
        Null space definition
    j : int < p
        The no of (from the first) rows of Null space W wrt which w is
        orthogonalized.
    Notes
    -----
    Assumes that W is orthogonal
    w changed in place
    """
    w -= np.dot(np.dot(w, W[:j].T), W[:j])
    return w


def _sym_decorrelation(W):
    """ Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = linalg.eigh(np.dot(W, W.T))
    # u (resp. s) contains the eigenvectors (resp. square roots of
    # the eigenvalues) of W * W.T
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)


def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """Deflationary FastICA using fun approx to neg-entropy function
    Used internally by FastICA.
    """

    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    n_iter = []

    # j is the index of the extracted component
    for j in range(n_components):
        w = w_init[j, :].copy()
        w /= np.sqrt((w ** 2).sum())

        for i in moves.xrange(max_iter):
            gwtx, g_wtx = g(fast_dot(w.T, X), fun_args)

            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w

            _gs_decorrelation(w1, W, j)

            w1 /= np.sqrt((w1 ** 2).sum())

            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break

        n_iter.append(i + 1)
        W[j, :] = w

    return W, max(n_iter)

def block_dot(X, Y, block_size, n_blocks=1):
    """Performs block-wise dot product

        |    |    |    |             |  Y1  |
        | X1 | X2 | X3 |  (n by bm)  |  Y2  | (bm by p)
        |    |    |    |             |  Y3  |

        | X1Y1 |
        | ---- |
    =   | X2Y2 |  (bn by p)
        | ---- |
        | X3Y3 |

    """
    if X.shape[1] != Y.shape[0]:
        raise ValueError("Matrix dimension mismatch")
    XY = np.zeros((X.shape[0]*n_blocks, Y.shape[1]))
    for b in range(0, n_blocks):
        Xblock = slice(b * X.shape[0], (b+1) * X.shape[0])
        Yblock = slice(b * block_size, (b+1) * block_size)
        XY[Xblock, :] = fast_dot(X[:,Yblock], Y[Yblock, :])
    return XY

def block_dot2(X, Y, block_size, n_blocks=1):
    """Performs block-wise dot product

        |    |    |    |             |    |    |    |
        | X1 | X2 | X3 |  (n by bm)  | Y1 | Y2 | Y3 |  (m by bp)
        |    |    |    |             |    |    |    |

        |      |      |      |
    =   | X1Y1 | X2Y2 | X3Y3 |  (n by bp)
        |      |      |      |

    """
    if block_size is None:
        block_size = Y.shape[1]
    if X.shape[1] != Y.shape[0] * n_blocks:
        raise ValueError("Matrix dimension mismatch")
    if Y.shape[1] != block_size * n_blocks:
        raise ValueError("Incorrect block dimensions")
    XY = np.zeros((X.shape[0], Y.shape[1]))
    for b in range(0, n_blocks):
        block = slice(b * block_size, (b+1) * block_size)
        cblock = slice(b * X.shape[0], (b+1) * X.shape[0])
        XY[:, block] = fast_dot(X[:,cblock], Y[:, block])
    return XY

def block_dot3(X, Y, block_sizeX, block_sizeY, n_blocks=1):
    """Performs block-wise dot product

        |  X1  |            |   Y1   |
        |  X2  | (bn by m)  |   Y2   | (bm by p)
        |  X3  |            |   Y3   |

        |  X1Y1  |
    =   |  X2Y2  |  (bn by p)
        |  X3Y3  |

    """
    if block_sizeX is None:
        block_sizeX = X.shape[0]
    if block_sizeY is None:
        block_sizeY = Y.shape[0]
    if X.shape[1] != block_sizeY:
        raise ValueError("Matrix dimension mismatch")
    if X.shape[0] != block_sizeX * n_blocks or Y.shape[0] != block_sizeY * n_blocks:
        raise ValueError("Incorrect block dimensions")
    XY = np.zeros((X.shape[0], Y.shape[1]))
    for b in range(0, n_blocks):
        Xblock = slice(b * block_sizeX, (b+1) * block_sizeX)
        Yblock = slice(b * block_sizeY, (b+1) * block_sizeY)
        XY[Xblock, :] = fast_dot(X[Xblock, :], Y[Yblock, :])
    return XY

# could remove block_size and infer from n_blocks
def block_transpose(X, block_size, n_blocks=1, axis=0):
    if block_size is None:
        if axis == 0:
            block_size = X.shape[1]
        else:
            block_size = X.shape[0]
    if axis == 0:
        Xt = np.zeros((X.shape[0]*n_blocks, block_size))
    else:
        Xt = np.zeros((block_size, X.shape[1]*n_blocks))
    for b in range(0, n_blocks):
        block = slice(b * block_size, (b+1) * block_size)
        if axis == 0:
            cblock = slice(b * X.shape[0], (b+1) * X.shape[0])
            Xt[cblock, :] = X[:, block]
        else:
            cblock = slice(b * X.shape[1], (b+1) * X.shape[1])
            Xt[:, cblock] = X[block, :]
    return Xt

# need to add "axis" argument
def block_inverse(X, block_size, n_blocks=1):
    """Performs block-wise pseudoinverse

        |    |    |    |
        | X1 | X2 | X3 |  (n by bm)
        |    |    |    |

        | X1^-1 |
        | ----- |
    ==> | X2^-1 |  (bm by n)
        | ----- |
        | X3^-1 |

    """
    if block_size is None:
        block_size = X.shape[1]
    if block_size * n_blocks != X.shape[1]:
        raise ValueError("Block size/number does not match matrix dimensions")
    X_inv = np.zeros((X.shape[1], X.shape[0]))
    for b in range(0, n_blocks):
        block = slice(b * block_size, (b+1) * block_size)
        X_inv[block, :] = linalg.pinv(X[:, block])
    return X_inv

def _ica_par(X, tol, g, fun_args, max_iter, w_init, block_size, n_blocks=1):
    """Parallel FastICA.
    Used internally by FastICA --main loop
    """
    # initialize and decorrelate each block in W
    W = np.zeros(w_init.shape)
    for b in range(0, n_blocks):
        block = slice(b * block_size, (b+1) * block_size) # saved slice for block b
        W[:, block] = _sym_decorrelation(w_init[:, block])
    del w_init
    p_ = float(X.shape[1])
    S = np.zeros((W.shape[0], X.shape[1]))
    for ii in moves.xrange(max_iter):
        S = fast_dot(W, X) #
        gwtx, g_wtx = g(S, fun_args)
        W1 = np.zeros(W.shape)
        for b in range(0, n_blocks):
            block = slice(b * block_size, (b+1) * block_size)
            Wb = W[:, block]
            Xb = X[block, :]
            W1[:, block] = _sym_decorrelation(fast_dot(gwtx, Xb.T) / p_
                                    - g_wtx[:, np.newaxis] * Wb)
        del gwtx, g_wtx
        converged = True;
        for b in range(0, n_blocks):
            block = slice(b * block_size, (b+1) * block_size)
            lim = max(abs(abs(np.diag(fast_dot(W1[:, block], W[:, block].T))) - 1))
            if lim >= tol:
                converged = False;
                break;
        W = W1
        if converged is True:
            break
    else:
        warnings.warn('FastICA did not converge.' +
                      ' You might want' +
                      ' to increase the number of iterations.')

    return W, ii + 1, S


# Some standard non-linear functions.
# XXX: these should be optimized, as they can be a bottleneck.
def _logcosh(x, fun_args=None):
    alpha = fun_args.get('alpha', 1.0)  # comment it out?

    x *= alpha
    gx = np.tanh(x, x)  # apply the tanh inplace
    g_x = np.empty(x.shape[0])
    # XXX compute in chunks to avoid extra allocation
    for i, gx_i in enumerate(gx):  # please don't vectorize.
        g_x[i] = (alpha * (1 - gx_i ** 2)).mean()
    return gx, g_x


def _exp(x, fun_args):
    exp = np.exp(-(x ** 2) / 2)
    gx = x * exp
    g_x = (1 - x ** 2) * exp
    return gx, g_x.mean(axis=-1)


def _cube(x, fun_args):
    return x ** 3, (3 * x ** 2).mean(axis=-1)


def fastica(X, n_components=None, algorithm="parallel", whiten=True,
            fun="logcosh", fun_args=None, max_iter=200, tol=1e-04, w_init=None,
            random_state=None, return_X_mean=False, compute_sources=True,
            return_n_iter=False, n_blocks=None, block_size=None):
    """Perform Fast Independent Component Analysis.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    n_components : int, optional
        Number of components to extract. If None no dimension reduction
        is performed.
    algorithm : {'parallel', 'deflation'}, optional
        Apply a parallel or deflational FASTICA algorithm.
    whiten : boolean, optional
        If True perform an initial whitening of the data.
        If False, the data is assumed to have already been
        preprocessed: it should be centered, normed and white.
        Otherwise you will get incorrect results.
        In this case the parameter n_components will be ignored.
    fun : string or function, optional. Default: 'logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. Example:
        def my_g(x):
            return x ** 3, 3 * x ** 2
    fun_args : dictionary, optional
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}
    max_iter : int, optional
        Maximum number of iterations to perform.
    tol: float, optional
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.
    w_init : (n_components, n_components) array, optional
        Initial un-mixing array of dimension (n.comp,n.comp).
        If None (default) then an array of normal r.v.'s is used.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.
    return_X_mean : bool, optional
        If True, X_mean is returned too.
    compute_sources : bool, optional
        If False, sources are not computed, but only the rotation matrix.
        This can save memory when working with big data. Defaults to True.
    return_n_iter : bool, optional
        Whether or not to return the number of iterations.
    Returns
    -------
    K : array, shape (n_components, n_features) | None.
        If whiten is 'True', K is the pre-whitening matrix that projects data
        onto the first n_components principal components. If whiten is 'False',
        K is 'None'.
    W : array, shape (n_components, n_components)
        Estimated un-mixing matrix.
        The mixing matrix can be obtained by::
            w = np.dot(W, K.T)
            A = w.T * (w * w.T).I
    S : array, shape (n_components, n_samples) | None
        Estimated source matrix
    X_mean : array, shape (n_features, )
        The mean over features. Returned only if return_X_mean is True.
    n_iter : int
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge. This is
        returned only when return_n_iter is set to `True`.
    Notes
    -----
    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = AS where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    un-mixing matrix W where ``S = W K X.``
    This implementation was originally made for data of shape
    [n_features, n_samples]. Now the input is transposed
    before the algorithm is applied. This makes it slightly
    faster for Fortran-ordered input.
    Implemented using FastICA:
    `A. Hyvarinen and E. Oja, Independent Component Analysis:
    Algorithms and Applications, Neural Networks, 13(4-5), 2000,
    pp. 411-430`
    """
    random_state = check_random_state(random_state)
    fun_args = {} if fun_args is None else fun_args
    # make interface compatible with other decompositions
    # a copy is required only for non whitened data
    X = check_array(X, copy=whiten).T

    alpha = fun_args.get('alpha', 1.0)
    if not 1 <= alpha <= 2:
        raise ValueError('alpha must be in [1,2]')

    if fun == 'logcosh':
        g = _logcosh
    elif fun == 'exp':
        g = _exp
    elif fun == 'cube':
        g = _cube
    elif callable(fun):
        def g(x, fun_args):
            return fun(x, **fun_args)
    else:
        exc = ValueError if isinstance(fun, six.string_types) else TypeError
        raise exc("Unknown function %r;"
                  " should be one of 'logcosh', 'exp', 'cube' or callable"
                  % fun)

    n, p = X.shape

    if not whiten and n_components is not None:
        # n_components = None
        warnings.warn('Ignoring n_components with whiten=False.')

    # if n_components is None:
    #     n_components = min(n, p)
    # if (n_components > min(n, p)):
    #     n_components = min(n, p)
    #     print("n_components is too large: it will be set to %s" % n_components)

    # If number of blocks and block size aren't set, default to 1 block
    if (n_blocks is None) or (block_size is None) or (n_blocks <= 1):
        n_blocks = 1
        block_size = X.shape[0]
    else:
        if (n_blocks * block_size) != X.shape[0]:
            raise ValueError('Number of blocks and block size do not match the'
                             'total number of elements.')

    if whiten:
        X1 = np.zeros((n_components*n_blocks, X.shape[1]))
        K = np.zeros((n_components, X.shape[0]))
        # X_mean = X.mean(axis=-1)
        # print("n_components:", n_components, "Xshape1:", X.shape[1])
        # K = np.zeros((n_components, X.shape[0]))
        # u, d, _ = linalg.svd(X, full_matrices=False)
        # del _
        # K = (u / d).T[:n_components] # see (6.33) p.140
        # X1 = block_dot(K, X, block_size, n_blocks)
        # X1 *= np.sqrt(p)

        for b in range(0, n_blocks):
            block = slice(b * block_size, (b+1) * block_size)
            cblock = slice(b * n_components, (b+1) * n_components)
            # Centering the columns (ie the variables)
            Xb = X[block, :]
            X_mean = Xb.mean(axis=-1)
            Xb -= X_mean[:, np.newaxis]

            # Whitening and preprocessing by PCA
            u, d, _ = linalg.svd(Xb, full_matrices=False)

            del _
            Kb = (u / d).T[:n_components] # see (6.33) p.140
            K[:, block] = Kb
            del u, d
            X1[cblock, :] = np.dot(Kb, Xb)
            # see (13.6) p.267 Here X1 is white and data
            # in X has been projected onto a subspace by PCA
        X1 *= np.sqrt(p)
        # Set block arguments based on size of blocks after whitening
        blockargs = {'n_blocks': n_blocks, 'block_size': n_components}
    else:
        # X must be casted to floats to avoid typing issues with numpy
        # 2.0 and the line below
        X1 = as_float_array(X, copy=False)  # copy has been taken care of
        blockargs = {'n_blocks': n_blocks, 'block_size': block_size}


    if w_init is None:
        w_init = np.asarray(random_state.normal(size=(n_components,
                            n_components * n_blocks)), dtype=X1.dtype)

    else:
        w_init = np.asarray(w_init)
        if w_init.shape != (n_components, n_components * n_blocks):
            raise ValueError('w_init has invalid shape -- should be %(shape)s'
                             % {'shape': (n_components, n_components * n_blocks)})

    kwargs = {'tol': tol,
              'g': g,
              'fun_args': fun_args,
              'max_iter': max_iter,
              'w_init': w_init}

    kwargs2 = dict(kwargs, **blockargs) # Combine kwargs with block settings

    if algorithm == 'parallel':
        W, n_iter, S = _ica_par(X1, **kwargs2)
    elif algorithm == 'deflation':
        W, n_iter = _ica_def(X1, **kwargs)
    else:
        raise ValueError('Invalid algorithm: must be either `parallel` or'
                         ' `deflation`.')
    del X1

    try:
        S
    except NameError:
        S = None

    if whiten:
        if compute_sources:
            S = fast_dot(W, block_dot(K, X, block_size, n_blocks)).T
        if return_X_mean:
            if return_n_iter:
                return K, W, S, X_mean, n_iter
            else:
                return K, W, S, X_mean
        else:
            if return_n_iter:
                return K, W, S, n_iter
            else:
                return K, W, S

    else:
        if compute_sources:
            S = fast_dot(W, X).T
        if return_X_mean:
            if return_n_iter:
                return None, W, S, None, n_iter
            else:
                return None, W, S, None
        else:
            if return_n_iter:
                return None, W, S, n_iter
            else:
                return None, W, S


class FastICA(BaseEstimator, TransformerMixin):
    """FastICA: a fast algorithm for Independent Component Analysis.
    Parameters
    ----------
    n_components : int, optional
        Number of components to use. If none is passed, all are used.
    algorithm : {'parallel', 'deflation'}
        Apply parallel or deflational algorithm for FastICA.
    whiten : boolean, optional
        If whiten is false, the data is already considered to be
        whitened, and no whitening is performed.
    fun : string or function, optional. Default: 'logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. Example:
        def my_g(x):
            return x ** 3, 3 * x ** 2
    fun_args : dictionary, optional
        Arguments to send to the functional form.
        If empty and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}.
    max_iter : int, optional
        Maximum number of iterations during fit.
    tol : float, optional
        Tolerance on update at each iteration.
    w_init : None of an (n_components, n_components) ndarray
        The mixing matrix to be used to initialize the algorithm.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.
    Attributes
    ----------
    components_ : 2D array, shape (n_components, n_features)
        The unmixing matrix.
    mixing_ : array, shape (n_features, n_components)
        The mixing matrix.
    n_iter_ : int
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge.
    Notes
    -----
    Implementation based on
    `A. Hyvarinen and E. Oja, Independent Component Analysis:
    Algorithms and Applications, Neural Networks, 13(4-5), 2000,
    pp. 411-430`
    """
    def __init__(self, n_components=None, algorithm='parallel', whiten=True,
                 fun='logcosh', fun_args=None, max_iter=200, tol=1e-4,
                 w_init=None, random_state=None, block_size=None, n_blocks=1):
        super(FastICA, self).__init__()
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state
        self.block_size = block_size
        self.n_blocks = n_blocks

    def _fit(self, X, compute_sources=False):
        """Fit the model
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        compute_sources : bool
            If False, sources are not computes but only the rotation matrix.
            This can save memory when working with big data. Defaults to False.
        Returns
        -------
            X_new : array-like, shape (n_samples, n_components)
        """
        fun_args = {} if self.fun_args is None else self.fun_args
        whitening, unmixing, sources, X_mean, self.n_iter_ = fastica(
            X=X, n_components=self.n_components, algorithm=self.algorithm,
            whiten=self.whiten, fun=self.fun, fun_args=fun_args,
            max_iter=self.max_iter, tol=self.tol, w_init=self.w_init,
            random_state=self.random_state, return_X_mean=True,
            compute_sources=compute_sources, return_n_iter=True,
            block_size=self.block_size, n_blocks=self.n_blocks)

        if self.whiten:
            # Kt = block_transpose(whitening, block_size=self.n_components, n_blocks=self.n_blocks, axis=1)
            self.components_ = block_dot2(unmixing, whitening, n_blocks=self.n_blocks, block_size=self.block_size)
            self.mean_ = X_mean
            self.whitening_ = whitening
        else:
            self.components_ = unmixing

        self.mixing_ = linalg.pinv(self.components_)

        if compute_sources:
            self.__sources = sources

        return sources

    def fit_transform(self, X, y=None):
        """Fit the model and recover the sources from X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        return self._fit(X, compute_sources=True)

    def fit(self, X, y=None):
        """Fit the model to X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        self
        """
        self._fit(X, compute_sources=False)
        return self

    def transform(self, X, y=None, copy=True):
        """Recover the sources from X (apply the unmixing matrix).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform, where n_samples is the number of samples
            and n_features is the number of features.
        copy : bool (optional)
            If False, data passed to fit are overwritten. Defaults to True.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        #check_is_fitted(self, 'mixing_')

        X = check_array(X, copy=copy)
        if self.whiten:
            X -= self.mean_

        return fast_dot(X, self.components_.T)

    def inverse_transform(self, X, copy=True):
        """Transform the sources back to the mixed data (apply mixing matrix).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Sources, where n_samples is the number of samples
            and n_components is the number of components.
        copy : bool (optional)
            If False, data passed to fit are overwritten. Defaults to True.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
        """
        #check_is_fitted(self, 'mixing_')

        if copy:
            X = X.copy()
        X = fast_dot(X, self.mixing_.T)
        if self.whiten:
            X += self.mean_

        return X
