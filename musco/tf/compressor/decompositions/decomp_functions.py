import numpy as np

def uv_decompose(matrix, max_rank=None, epsilon=1e-8):
    """
    Calcultes a skeleton decomposition using SVD
    Factor U takes left singular vectors multiplied by singular values,
    factor V takes right singular vectors.
    Parameters
    ----------
    matrix: np.array, matrix to decompose
    max_rank: int, maximal value of rank (included)
    epsilon: maximal difference in Frobenius norm
             of the resulting decomposition
    Return
    ------
    uv: tuple, two factors of the decomposition
    """
    # import pdb; pdb.set_trace()
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    errors = np.sqrt(np.abs(np.sum(s**2) - np.cumsum(s**2)))
    rank_num = np.argmax(errors < epsilon)
    if rank_num == 0:  # none of the errors < epsilon => full rank
        rank_num = s.shape[0]
    if max_rank is None:
        max_rank = s.shape[0]
    rank = min(max_rank, rank_num)
    return np.dot(u[:, :rank+1], np.diag(s[:rank+1])), v[:rank+1, :]