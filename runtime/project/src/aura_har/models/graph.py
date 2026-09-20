from __future__ import annotations

import numpy as np

NTU_INWARD_1BASED = [
    (1, 2),
    (2, 21),
    (3, 21),
    (4, 3),
    (5, 21),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 21),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
    (22, 23),
    (23, 8),
    (24, 25),
    (25, 12),
]


def edge_to_matrix(edges: list[tuple[int, int]], num_nodes: int) -> np.ndarray:
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for source, target in edges:
        matrix[target, source] = 1.0
    return matrix


def normalize_digraph(matrix: np.ndarray) -> np.ndarray:
    degree = matrix.sum(axis=0)
    inverse = np.zeros_like(degree)
    inverse[degree > 0] = 1.0 / degree[degree > 0]
    return matrix @ np.diag(inverse)


def ntu_adjacency(num_nodes: int = 25) -> np.ndarray:
    if num_nodes != 25:
        raise ValueError("The bundled NTU graph requires exactly 25 joints")
    inward = [(source - 1, target - 1) for source, target in NTU_INWARD_1BASED]
    outward = [(target, source) for source, target in inward]
    identity = edge_to_matrix([(index, index) for index in range(num_nodes)], num_nodes)
    return np.stack(
        [
            identity,
            normalize_digraph(edge_to_matrix(inward, num_nodes)),
            normalize_digraph(edge_to_matrix(outward, num_nodes)),
        ]
    ).astype(np.float32)
