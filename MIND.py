import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
import itertools
from tqdm import tqdm
import pandas as pd # Assuming helpers use pandas

# Assuming MIND_helpers.py is in the same directory or accessible
from MIND_helpers import get_verts

# Import the new OT metrics and similarity conversion
from ot_metrics import (
    dissimilarity_to_sim,
    compute_emd_dissimilarity,
    compute_sinkhorn_dissimilarity,
    compute_sw_dissimilarity,
    compute_max_sw_dissimilarity
)

# --- get_kl function (元のコードのまま) ---
def get_kl(v1, v2):
    """
    Estimates KL-divergence KL(v1 || v2) using k-NN (k=1).
    v1, v2 are numpy arrays with shape (N_vertices, N_features).
    Returns D = KL(v1 || v2). Calculation follows Wang et al. 2009 SciPy Proc.
    Note: function assumes N_features > 0. If N_features = 0, returns np.inf.
    """
    n, d = v1.shape
    m, d = v2.shape

    if d == 0:
        # If no features, KL is ill-defined. Return inf? Or 0 if n==m?
        # MIND paper likely assumes d > 0 based on features used. Let's return inf.
        return np.inf

    # Need at least k+1 points to find k non-self neighbors. Here k=1. Need >= 2 points.
    if n < 2 or m < 1: # Need at least 2 in v1 for nn (excluding self), 1 in v2 for nn search
        # Returning inf seems appropriate as KL estimation is unreliable/impossible
        print(f"Warning: Insufficient points for KL estimation (n={n}, m={m}). Returning inf.")
        return np.inf

    # Fit KNN models
    try:
        # For rho (distance in v1): find k=1 nearest neighbor excluding self (so use k=2)
        nbrs1 = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='sqeuclidean').fit(v1)
        # For nu (distance in v2): find k=1 nearest neighbor
        nbrs2 = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='sqeuclidean').fit(v2)

        # Distances: rho = dist to NN in v1 (excluding self), nu = dist to NN in v2
        rho, _ = nbrs1.kneighbors(v1)
        # Check if any point is identical (rho[:, 1] might be 0)
        # If n > 1, rho[:, 1] corresponds to the nearest distinct point's distance.
        # If n == 1, this index is out of bounds, but we handled n<2 earlier.
        rho = rho[:, 1]

        nu, _ = nbrs2.kneighbors(v1)
        nu = nu[:, 0] # k=1, so index 0 is the nearest neighbor in v2

        # Ensure distances are non-negative and handle potential zeros for division stability
        rho = np.maximum(rho, np.finfo(float).eps)
        nu = np.maximum(nu, np.finfo(float).eps)

        # Estimate KL divergence
        # Formula from Perez-Cruz, F. (2008). Estimation of Information Theoretic Measures for Continuous Random Variables. NIPS.
        # (Adapting Kozachenko-Leonenko estimator, related to kNN entropy estimation)
        # Note: MIND paper cites Wang et al. 2009 which likely uses a similar estimator.
        # Original MIND code seems to implement: d/n * sum(log(nu_i / rho_i)) + log(m / (n - 1))
        # Let's stick to the original code's implementation:
        kl_est = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1.0))


        # Clip at 0, assuming negative estimates imply KL near zero
        return max(0.0, kl_est)

    except Exception as e:
        print(f"Warning: k-NN KL computation failed. Returning inf. Error: {e}")
        return np.inf

# --- dissimilarity_to_sim function (from ot_metrics.py, can be defined here or imported) ---
# Included here for completeness if ot_metrics.py is not used, but prefer import
# def dissimilarity_to_sim(D):
#     if D < 0:
#         raise ValueError("Dissimilarity D must be non-negative.")
#     return 1.0 / (1.0 + D + np.finfo(float).eps)


def get_mind_similarity(df, sub, hemi, rois, feats,
                        method='mind', # Added method parameter
                        **kwargs):      # Added kwargs for method parameters
    """
    Calculate the MIND or OT-based similarity matrix for a given subject and hemisphere.
    Uses the original get_kl function for the 'mind' method.

    Args:
        df (pd.DataFrame): DataFrame containing all vertex data.
        sub (str): Subject ID.
        hemi (str): Hemisphere ('lh' or 'rh').
        rois (list): List of ROI names.
        feats (list): List of feature names.
        method (str): Similarity method ('mind', 'emd', 'sinkhorn', 'sw', 'max-sw').
                      Defaults to 'mind'.
        **kwargs: Additional keyword arguments for the chosen method:
                  - reg_epsilon (float): For 'sinkhorn'. Heuristic if None.
                  - n_projections (int): For 'sw' and 'max-sw'. Defaults to 100.
                  - p (int): Order for SW/max-SW distances. Defaults to 2.
                  - Other kwargs passed to the respective POT function.

    Returns:
        np.ndarray: A symmetric similarity matrix (len(rois) x len(rois)).
                    Diagonal elements are 1. Off-diagonal elements in (0, 1].
                    Returns None if data retrieval fails for any ROI.
    """
    n_rois = len(rois)
    sim_matrix = np.zeros((n_rois, n_rois))

    # Retrieve vertex data for all ROIs first
    roi_verts = {}
    for roi in rois:
        verts = get_verts(df, sub, hemi, roi, feats)
        if verts is None or verts.shape[0] == 0:
            print(f"Warning: No vertices found for {sub} {hemi} {roi}. Cannot compute similarities.")
            return None
        # Ensure data has features
        if verts.ndim == 1: # If only one vertex, reshape
             verts = verts.reshape(1, -1)
        if verts.shape[1] == 0 and method=='mind':
             print(f"Warning: Zero features found for {sub} {hemi} {roi}. MIND requires features.")
             # MIND's get_kl handles d=0, returning inf. OT methods might handle differently.
             # We let get_kl handle it for 'mind'. For OT, they might return 0 or inf.
        roi_verts[roi] = verts


    # Calculate pairwise similarities
    for i in range(n_rois):
        for j in range(i, n_rois):
            if i == j:
                sim_matrix[i, j] = 1.0
                continue

            roi1 = rois[i]
            roi2 = rois[j]
            v1 = roi_verts[roi1]
            v2 = roi_verts[roi2]

            dissimilarity = np.inf # Default to inf

            try:
                # --- MIND Calculation: Use original get_kl and symmetrize ---
                if method == 'mind':
                    kl12 = get_kl(v1, v2) # Use the original MIND function
                    kl21 = get_kl(v2, v1) # Use the original MIND function
                    if np.isinf(kl12) or np.isinf(kl21):
                        dissimilarity = np.inf
                        print(f"Warning: KL divergence is infinite between {roi1} and {roi2}. Setting dissimilarity to inf.")
                    else:
                        # Symmetrize KL divergence as in original MIND approach
                        dissimilarity = (kl12 + kl21) / 2.0
                # --- OT Calculations: Use functions from ot_metrics.py ---
                elif method == 'emd':
                    dissimilarity = compute_emd_dissimilarity(v1, v2, **kwargs)
                elif method == 'sinkhorn':
                    reg_epsilon = kwargs.get('reg_epsilon', None)
                    ot_kwargs = {k: v for k, v in kwargs.items() if k != 'reg_epsilon'}
                    dissimilarity = compute_sinkhorn_dissimilarity(v1, v2, reg_epsilon=reg_epsilon, **ot_kwargs)
                elif method == 'sw':
                    n_projections = kwargs.get('n_projections', 100)
                    p_order = kwargs.get('p', 2) # Renamed to avoid clash with feature dimension 'p'
                    ot_kwargs = {k: v for k, v in kwargs.items() if k not in ['n_projections', 'p']}
                    dissimilarity = compute_sw_dissimilarity(v1, v2, n_projections=n_projections, p=p_order, **ot_kwargs)
                elif method == 'max-sw':
                    n_projections = kwargs.get('n_projections', 100)
                    p_order = kwargs.get('p', 2) # Renamed to avoid clash with feature dimension 'p'
                    ot_kwargs = {k: v for k, v in kwargs.items() if k not in ['n_projections', 'p']}
                    dissimilarity = compute_max_sw_dissimilarity(v1, v2, n_projections=n_projections, p=p_order, **ot_kwargs)
                else:
                    raise ValueError(f"Unknown similarity method: {method}")

                # --- Convert dissimilarity to similarity (Common Step) ---
                if np.isinf(dissimilarity) or np.isnan(dissimilarity):
                    similarity = 0.0 # Assign 0 similarity if distance is infinite or NaN
                    if not np.isinf(dissimilarity): # Only print warning for NaN, inf handled above
                         print(f"Warning: Dissimilarity is NaN between {roi1} and {roi2}. Setting similarity to 0.")
                elif dissimilarity < 0:
                     print(f"Warning: Negative dissimilarity ({dissimilarity:.2f}) computed between {roi1} and {roi2} using {method}. Clipping to 0.")
                     similarity = dissimilarity_to_sim(0.0) # Should be 1.0
                else:
                    similarity = dissimilarity_to_sim(dissimilarity)

            except Exception as e:
                print(f"Error computing {method} similarity between {roi1} and {roi2}: {e}")
                similarity = 0.0 # Assign 0 similarity on general error

            sim_matrix[i, j] = similarity
            sim_matrix[j, i] = similarity # Ensure symmetry

    return sim_matrix


# --- get_mind_output remains the same as previous proposal ---
def get_mind_output(args):
    """
    Wrapper function for parallel processing. Calculates the similarity matrix.
    """
    df, sub, hemi, rois, feats, method, kwargs = args # Unpack arguments
    try:
        output = get_mind_similarity(df, sub, hemi, rois, feats, method=method, **kwargs)
        return output
    except Exception as e:
        print(f"Error processing {sub} {hemi} with method {method}: {e}")
        return None # Return None on failure within the worker


# --- Example Usage (similar to original, but adding method) ---
if __name__ == '__main__':

    # This part requires the actual data loading and setup from the notebooks
    # For demonstration purposes, let's assume 'df', 'subs', 'hemis', 'rois', 'feats' are loaded

    # Example Placeholder Data (replace with actual loading)
    # df = pd.read_csv(...)
    # subs = ['sub-01', 'sub-02']
    # hemis = ['lh', 'rh']
    # rois = ['roi_A', 'roi_B', 'roi_C']
    # feats = ['feature1', 'feature2']

    # --- Parameters ---
    # selected_method = 'mind'      # Or 'emd', 'sinkhorn', 'sw', 'max-sw'
    # method_params = {}            # For EMD or MIND
    # method_params = {'reg_epsilon': 0.1} # For Sinkhorn
    # method_params = {'n_projections': 200} # For SW or max-SW
    # n_cores = 4                   # Number of cores for parallel processing

    # print(f"Calculating similarity using method: {selected_method}")
    # print(f"Using parameters: {method_params}")

    # --- Prepare arguments for parallel processing ---
    # args_list = []
    # for sub in subs:
    #     for hemi in hemis:
    #         args_list.append((df, sub, hemi, rois, feats, selected_method, method_params))

    # --- Run in parallel ---
    # with Pool(processes=n_cores) as pool:
    #     results = list(tqdm(pool.imap(get_mind_output, args_list), total=len(args_list)))

    # --- Process results ---
    # output_matrices = {}
    # failed_count = 0
    # for i, result in enumerate(results):
    #     sub, hemi = args_list[i][1], args_list[i][2]
    #     if result is not None:
    #         output_matrices[(sub, hemi)] = result
    #     else:
    #         print(f"Failed to compute similarity for {sub} {hemi}")
    #         failed_count += 1

    # print(f"\nSuccessfully computed {len(output_matrices)} matrices.")
    # if failed_count > 0:
    #     print(f"Failed to compute {failed_count} matrices.")

    # Now output_matrices dictionary holds the similarity matrices
    # Example: Access matrix for sub-01, lh: output_matrices[('sub-01', 'lh')]

    pass # Placeholder to avoid syntax error if example code is commented out
