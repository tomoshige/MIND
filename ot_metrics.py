# ot_metrics.py

import numpy as np
import ot
from scipy.spatial.distance import cdist # For epsilon heuristic

def dissimilarity_to_sim(D):
    """
    Converts a dissimilarity measure (like squared distance or KL divergence)
    to a similarity measure in the range (0, 1].

    Args:
        D (float): The dissimilarity value (>= 0).

    Returns:
        float: The similarity value.
    """
    if D < 0:
        raise ValueError("Dissimilarity D must be non-negative.")
    # Add a small epsilon to prevent division by zero if D is exactly 0
    # Although mathematically D=0 maps to S=1, float precision might cause issues.
    # Alternatively, handle D=0 explicitly: return 1.0 if D == 0 else 1.0 / (1.0 + D)
    return 1.0 / (1.0 + D + np.finfo(float).eps)

def compute_emd_dissimilarity(v1, v2, **ot_kwargs):
    """
    Computes the squared Earth Mover's Distance (Wasserstein-2 distance)
    between two empirical distributions represented by vertex data.

    Args:
        v1 (np.ndarray): Vertex data for region 1 (n_vertices1, n_features).
        v2 (np.ndarray): Vertex data for region 2 (n_vertices2, n_features).
        **ot_kwargs: Additional keyword arguments passed to ot.emd2.

    Returns:
        float: The squared EMD (W_2^2). Returns np.inf if calculation fails.
    """
    n1 = v1.shape[0]
    n2 = v2.shape[0]
    if n1 == 0 or n2 == 0:
        return np.inf # Cannot compute distance with empty sets

    # Uniform weights for empirical distributions
    a = ot.unif(n1)
    b = ot.unif(n2)

    # Cost matrix: squared Euclidean distance
    M = cdist(v1, v2, metric='sqeuclidean')

    try:
        # ot.emd2 returns the total cost (which is W_2^2 for sqeuclidean)
        # For empirical distributions with uniform weights, the cost is sum(Pi_ij * M_ij)
        # where sum(Pi_ij) = 1. POT returns sum(Pi_ij * M_ij),
        # which is already the squared distance we want for W_2^2.
        squared_distance = ot.emd2(a, b, M, **ot_kwargs)
        # Ensure non-negativity due to potential numerical issues
        return max(0.0, squared_distance)
    except Exception as e:
        print(f"Warning: EMD computation failed between regions. Returning inf. Error: {e}")
        return np.inf


def _default_sinkhorn_epsilon(v1, v2):
    """Heuristic for choosing the Sinkhorn regularization parameter epsilon."""
    # Use 5% of the median squared Euclidean distance within the combined data
    # or within one of the point clouds if the other is empty (though that case returns inf earlier)
    if v1.shape[0] > 1:
        combined_v = v1
    elif v2.shape[0] > 1:
        combined_v = v2
    else: # If both have <= 1 point, default epsilon is small
        return 0.01

    if combined_v.shape[0] > 1:
      dist_matrix = cdist(combined_v, combined_v, metric='sqeuclidean')
      # Use non-zero distances to calculate median
      non_zero_dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
      if len(non_zero_dists) > 0:
          median_sqdist = np.median(non_zero_dists)
          return 0.05 * median_sqdist
      else: # Only one unique point
          return 0.01
    else: # Only one point in total
        return 0.01


def compute_sinkhorn_dissimilarity(v1, v2, reg_epsilon=None, **ot_kwargs):
    """
    Computes the squared Sinkhorn distance (approximated regularized OT)
    between two empirical distributions.

    Args:
        v1 (np.ndarray): Vertex data for region 1 (n_vertices1, n_features).
        v2 (np.ndarray): Vertex data for region 2 (n_vertices2, n_features).
        reg_epsilon (float, optional): Regularization parameter epsilon.
                                       If None, a heuristic value is used.
        **ot_kwargs: Additional keyword arguments passed to ot.sinkhorn2.

    Returns:
        float: The squared Sinkhorn distance. Returns np.inf if calculation fails.
    """
    n1 = v1.shape[0]
    n2 = v2.shape[0]
    if n1 == 0 or n2 == 0:
        return np.inf

    a = ot.unif(n1)
    b = ot.unif(n2)
    M = cdist(v1, v2, metric='sqeuclidean')

    if reg_epsilon is None:
        reg_epsilon = _default_sinkhorn_epsilon(v1, v2)
        # Ensure epsilon is positive
        reg_epsilon = max(reg_epsilon, 1e-6)


    try:
        # ot.sinkhorn2 returns the regularized transport cost (approx W_2^2)
        squared_distance = ot.sinkhorn2(a, b, M, reg=reg_epsilon, **ot_kwargs)[0] # [0] to get cost
        return max(0.0, squared_distance)
    except Exception as e:
        print(f"Warning: Sinkhorn computation failed. Epsilon={reg_epsilon}. Returning inf. Error: {e}")
        return np.inf

def compute_sw_dissimilarity(v1, v2, n_projections=100, p=2, **ot_kwargs):
    """
    Computes the squared Sliced Wasserstein distance between two empirical
    distributions.

    Args:
        v1 (np.ndarray): Vertex data for region 1 (n_vertices1, n_features).
        v2 (np.ndarray): Vertex data for region 2 (n_vertices2, n_features).
        n_projections (int): Number of random projections to use.
        p (int): The order p for W_p distance (default is 2).
        **ot_kwargs: Additional keyword arguments passed to
                     ot.sliced_wasserstein_distance.

    Returns:
        float: The squared SW distance (SW_p^2). Returns np.inf if calculation fails.
    """
    n1 = v1.shape[0]
    n2 = v2.shape[0]
    if n1 == 0 or n2 == 0:
        return np.inf

    try:
        # ot.sliced_wasserstein_distance returns SW_p distance
        distance = ot.sliced_wasserstein_distance(v1, v2, n_projections=n_projections, p=p, **ot_kwargs)
        return max(0.0, distance**p) # Return SW_p^p (squared distance for p=2)
    except Exception as e:
        print(f"Warning: SW computation failed. Returning inf. Error: {e}")
        return np.inf

def compute_max_sw_dissimilarity(v1, v2, n_projections=100, p=2, **ot_kwargs):
    """
    Computes the squared Max-Sliced Wasserstein distance between two empirical
    distributions.

    Args:
        v1 (np.ndarray): Vertex data for region 1 (n_vertices1, n_features).
        v2 (np.ndarray): Vertex data for region 2 (n_vertices2, n_features).
        n_projections (int): Number of projections to initialize optimization / check.
                             Actual implementation might use gradient ascent. POT uses
                             optimization initialized with random projections.
        p (int): The order p for W_p distance (default is 2).
        **ot_kwargs: Additional keyword arguments passed to
                     ot.max_sliced_wasserstein_distance.

    Returns:
        float: The squared max-SW distance (max-SW_p^2). Returns np.inf if calculation fails.
    """
    n1 = v1.shape[0]
    n2 = v2.shape[0]
    if n1 == 0 or n2 == 0:
        return np.inf
    # Ensure features dimension > 0
    if v1.shape[1] == 0:
        return 0.0 if n1==n2 else np.inf # Convention: 0 dist if same # points, else inf


    try:
        # ot.max_sliced_wasserstein_distance returns max-SW_p distance
        distance = ot.max_sliced_wasserstein_distance(v1, v2, n_projections=n_projections, p=p, **ot_kwargs)
        return max(0.0, distance**p) # Return max-SW_p^p (squared distance for p=2)
    except Exception as e:
        # Catch potential issues, e.g., low-rank matrices if p > d sometimes causes issues
        print(f"Warning: max-SW computation failed. Returning inf. Error: {e}")
        return np.inf
