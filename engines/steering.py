import numpy as np
from sklearn.decomposition import PCA


class VectorSteering:
    """Subspace Orthogonal Projection, Vector Momentum & Topological Steering."""

    @staticmethod
    def compute_pca_subspace(history_vecs, k=3):
        if len(history_vecs) < k + 1:
            return None
        arr = np.array(history_vecs[-25:], dtype=np.float64)
        n_comp = min(k, arr.shape[0], arr.shape[1])
        if n_comp < 1:
            return None
        pca = PCA(n_components=n_comp)
        pca.fit(arr)
        return pca.components_

    @staticmethod
    def orthogonal_projection(candidate_vec, components):
        if components is None or len(components) == 0:
            return candidate_vec
        v_ortho = np.array(candidate_vec, dtype=np.float64).copy()
        for comp in components:
            proj = float(np.dot(v_ortho, comp))
            v_ortho -= proj * comp
        norm = np.linalg.norm(v_ortho)
        if norm > 1e-6:
            v_ortho /= norm
        return v_ortho

    @staticmethod
    def topological_steering(candidate_vec, cohom_data, push_weight=0.5):
        """Pushes candidate vector away from 1-cycle loops and towards 2-void centroids."""
        if cohom_data is None:
            return candidate_vec
        v_steered = np.array(candidate_vec, dtype=np.float64).copy()

        # Loop Repulsion: push orthogonal to repetitive 1-cycle generator
        cycle_dir = cohom_data.get("cycle_direction")
        if cycle_dir is not None:
            cycle_proj = float(np.dot(v_steered, cycle_dir))
            v_steered -= push_weight * cycle_proj * cycle_dir

        # Void Centroid Pull: steer into genuine non-bounding 2-void cavity
        void_cent = cohom_data.get("void_centroid")
        if void_cent is not None:
            dir_to_void = void_cent - v_steered
            norm = np.linalg.norm(dir_to_void)
            if norm > 1e-6:
                v_steered += (push_weight * 0.4) * (dir_to_void / norm)

        norm = np.linalg.norm(v_steered)
        if norm > 1e-6:
            v_steered /= norm
        return v_steered

    @staticmethod
    def compute_momentum_vector(gold_vecs, baseline_vec):
        if not gold_vecs:
            return None
        gold_arr = np.array(gold_vecs, dtype=np.float64)
        mean_gold = np.mean(gold_arr, axis=0)
        direction = mean_gold - baseline_vec
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        return direction

    @staticmethod
    def apply_momentum_steering(candidate_vec, gold_vecs, momentum_weight=0.3):
        """Blends high-value historical trajectory momentum into candidate vector."""
        if not gold_vecs or momentum_weight <= 0:
            return candidate_vec
        momentum = VectorSteering.compute_momentum_vector(gold_vecs, candidate_vec)
        if momentum is None:
            return candidate_vec
        v_blended = (1.0 - momentum_weight) * candidate_vec + momentum_weight * momentum
        norm = np.linalg.norm(v_blended)
        return v_blended / norm if norm > 1e-6 else candidate_vec

    @staticmethod
    def derive_semantic_contrast_themes(steered_vec, history_vecs, history_concepts, top_k=2):
        """
        Calculates semantic alignment of the steered vector against recent concepts,
        identifying the concepts of highest repulsion and maximal orthogonal distance.
        """
        if not history_vecs or not history_concepts:
            return []
        
        recent_vecs = np.array(history_vecs[-15:], dtype=np.float64)
        recent_concepts = history_concepts[-15:]
        
        norm_s = np.linalg.norm(steered_vec) + 1e-9
        s_unit = steered_vec / norm_s
        
        norm_h = np.linalg.norm(recent_vecs, axis=1, keepdims=True) + 1e-9
        h_unit = recent_vecs / norm_h
        
        sims = np.dot(h_unit, s_unit)
        # Find concepts with lowest similarity to the steered vector (most orthogonal)
        orthogonal_indices = np.argsort(sims)[:top_k]
        return [recent_concepts[idx] for idx in orthogonal_indices if idx < len(recent_concepts)]
