import numpy as np
import pandas as pd


def swap_r(mat, i, j):
    """swap rows within a matrix.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('input 2d-tensor is not a square.')
    if i >= mat.shape[0]:
        raise IndexError('index out of bound.')
    if j >= mat.shape[0]:
        raise IndexError('index out of bound.')

    mat[i, :], mat[j, :] = mat[j, :], mat[i, :]
    return mat


def swap_c(mat, i, j):
    """swap columns within a matrix.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('input 2d-tensor is not a square.')
    if i >= mat.shape[1]:
        raise IndexError('index out of bound.')
    if j >= mat.shape[1]:
        raise IndexError('index out of bound.')

    mat[:, i], mat[:, j] = mat[:, j], mat[:, i]
    return mat


class MatMetric:
    def __init__(self, mat):
        """store matrix, and calculte differences of upper triangle.

        Args:
            mat: numpy.ndarray, a 2d-square matrix.

        Raises:
            ValueError: when input matrix is not square.
        """
        self._mat = mat
        self._nrows, self._ncols = mat.shape
        if self._nrows != self._ncols:
            raise ValueError("Input tensor is not a square matrix.")
        self._obj = self.evaluate()

    @property
    def nrows(self):
        return self._nrows

    @property
    def ncols(self):
        return self._ncols
        
    def evaluate(self):
        diff_per_row = []
        for i in range(self._nrows):
            this_diff = np.diff(self._mat[i, i:])





def getSymMatOrdering(mat, threshold=1e-6, n_iter=100):
    """get ordered indices that cluster similar values in a matrix together.

    Args:
        mat: numpy array, a symmetric matrix input.
        threshold: float, below which an iteration stops.
        n_iter: int, upper limit of number of iterations.

    Returns:
        ordered_indices: the final indices.

    Raises:
        ValueError: mat is not a square 2d-tensor.
    """
    if type(mat) == list:
        mat = np.array(mat)     # in case input is a list
    elif type(mat) == pd.core.frame.DataFrame:
        mat = mat.values        # in case input is a pandas.dataframe

    nrows, ncols = mat.shape
    if nrows != ncols:
        raise ValueError('input 2d-tensor is not a square.')

    metric = MatMetric(mat)
    objective = metric.evaluate()
    diff = objective
    counter = 0
    while (np.abs(diff) > threshold) and (counter < n_iter):
        metric.optimize()
        new_obj = metric.evaluate()
        diff = new_obj - objective
        objective = new_obj
        counter += 1

    return metric.answer()
