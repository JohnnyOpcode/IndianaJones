import numpy as np
from sklearn.decomposition import PCA


class VectorSteering:
    """Subspace Orthogonal Projection, Vector Momentum & Topological Steering."""

    @staticmethod
    def compute_pca_subspace(history_vecs, k=2):
        if len(history_vecs) < k + 1:
            return None
        arr = np.array(history_vecs[-20:])
        pca = PCA(n_components=min(k, arr.shape[0], arr.shape[1]))
        pca.fit(arr)
        return pca.components_

    @staticmethod
    def orthogonal_projection(candidate_vec, components):
        if components is None:
            return candidate_vec
        v_ortho = candidate_vec.copy()
        for comp in components:
            v_ortho -= np.dot(v_ortho, comp) * comp
        norm = np.linalg.norm(v_ortho)
        if norm > 1e-6:
            v_ortho /= norm
        return v_ortho

    @staticmethod
    def topological_steering(candidate_vec, cohom_data, push_weight=0.5):
        """Pushes candidate vector away from 1-cycle loops and towards 2-void centroids."""
        if cohom_data is None:
            return candidate_vec
        v_steered = candidate_vec.copy()

        # Loop Repulsion (if 1-cycle present)
        cycle_dir = cohom_data.get("cycle_direction")
        if cycle_dir is not None:
            v_steered -= push_weight * np.dot(v_steered, cycle_dir) * cycle_dir

        # Void Centroid Pull (if 2-void present)
        void_cent = cohom_data.get("void_centroid")
        if void_cent is not None:
            dir_to_void = void_cent - v_steered
            norm = np.linalg.norm(dir_to_void)
            if norm > 1e-6:
                v_steered += (push_weight * 0.5) * (dir_to_void / norm)

        norm = np.linalg.norm(v_steered)
        if norm > 1e-6:
            v_steered /= norm
        return v_steered

    @staticmethod
    def compute_momentum_vector(gold_vecs, baseline_vec):
        if not gold_vecs:
            return None
        gold_arr = np.array(gold_vecs)
        mean_gold = np.mean(gold_arr, axis=0)
        direction = mean_gold - baseline_vec
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        return direction
