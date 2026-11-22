import numpy as np


def is_symmetric(adjacency):
    return bool(np.allclose(adjacency, adjacency.T))


def remove_self_loops(adjacency):
    cleaned = adjacency.copy()
    np.fill_diagonal(cleaned, 0)
    return cleaned


def symmetric_normalise(adjacency):
    # Computes D^(-1/2) (A + I) D^(-1/2) where D is the degree matrix of A + I
    with_loops = remove_self_loops(adjacency) + np.eye(adjacency.shape[0])
    degree = with_loops.sum(axis = 1)
    inverse_root = degree ** -0.5
    return inverse_root[:, None] * with_loops * inverse_root[None, :]


def row_normalise(adjacency):
    # Computes D^(-1) (A + I) so that every row sums to 1
    with_loops = remove_self_loops(adjacency) + np.eye(adjacency.shape[0])
    degree = with_loops.sum(axis = 1)
    return with_loops / degree[:, None]


def normalise_fixed_adjacency(adjacency):
    # The symmetric form is only valid for a symmetric graph, so a directed graph gets the row form
    if is_symmetric(adjacency):
        return symmetric_normalise(adjacency), "symmetric"
    return row_normalise(adjacency), "row"