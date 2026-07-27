import numpy as np


class CohomologyEngine:
    """Persistent Cohomology and Sheaf Obstruction Evaluator over Latent Embeddings."""

    def __init__(self, max_points=25, eps_percentile=50):
        self.max_points = max_points
        self.eps_percentile = eps_percentile

    def compute_simplicial_cohomology(self, history_vecs):
        """
        Builds a Vietoris-Rips simplicial complex over history_vecs,
        calculates boundary matrices d0, d1, d2 over R, and derives Betti numbers (b0, b1, b2),
        cohomological dimension (cd), and topological cycle/void guidance.
        """
        if not history_vecs or len(history_vecs) < 3:
            return {
                "b0": 1,
                "b1": 0,
                "b2": 0,
                "cohomological_dim": 1.0,
                "sheaf_obstruction": 0.0,
                "cohomological_novelty": 0.5,
                "cycle_direction": None,
                "void_centroid": None,
            }

        arr = np.array(history_vecs[-self.max_points:])
        N = len(arr)

        # Pairwise distance matrix
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(arr[i] - arr[j])
                dists[i, j] = dists[j, i] = d

        non_zero_dists = dists[dists > 1e-6]
        if len(non_zero_dists) == 0:
            epsilon = 0.5
        else:
            epsilon = float(np.percentile(non_zero_dists, self.eps_percentile))

        # 0-simplices C0: nodes 0..N-1
        c0 = list(range(N))

        # 1-simplices C1: pairs (i, j) with dist <= epsilon
        c1 = []
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] <= epsilon:
                    c1.append((i, j))

        # 2-simplices C2: triples (i, j, k) with all dists <= epsilon
        c2 = []
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] <= epsilon:
                    for k in range(j + 1, N):
                        if dists[i, k] <= epsilon and dists[j, k] <= epsilon:
                            c2.append((i, j, k))

        # Boundary operator partial_1: C1 -> C0
        num_c0 = len(c0)
        num_c1 = len(c1)
        num_c2 = len(c2)

        if num_c1 > 0:
            B1 = np.zeros((num_c0, num_c1))
            for idx, (i, j) in enumerate(c1):
                B1[j, idx] = 1.0
                B1[i, idx] = -1.0
            rank_B1 = int(np.linalg.matrix_rank(B1))
        else:
            rank_B1 = 0

        if num_c2 > 0:
            B2 = np.zeros((num_c1, num_c2))
            c1_index = {edge: idx for idx, edge in enumerate(c1)}
            for idx, (i, j, k) in enumerate(c2):
                if (j, k) in c1_index:
                    B2[c1_index[(j, k)], idx] += 1.0
                if (i, k) in c1_index:
                    B2[c1_index[(i, k)], idx] -= 1.0
                if (i, j) in c1_index:
                    B2[c1_index[(i, j)], idx] += 1.0
            rank_B2 = int(np.linalg.matrix_rank(B2))
        else:
            rank_B2 = 0

        # Betti numbers
        b0 = max(1, num_c0 - rank_B1)
        b1 = max(0, num_c1 - rank_B2 - rank_B1)
        b2 = max(0, num_c2 - rank_B2) if num_c2 > 0 else 0

        # Cohomological dimension
        cd = float(b0 + 1.5 * b1 + 2.0 * b2)

        # Sheaf Obstruction metric over last vectors (coboundary sections)
        if N >= 3:
            sec_diffs = arr[1:] - arr[:-1]
            sec_coboundary = sec_diffs[1:] - sec_diffs[:-1]
            norm_sec = np.linalg.norm(sec_diffs) + 1e-6
            sheaf_obstruction = float(np.linalg.norm(sec_coboundary) / norm_sec)
        else:
            sheaf_obstruction = 0.0

        # Cycle direction if b1 > 0
        cycle_direction = None
        if b1 > 0 and num_c1 > 0:
            cycle_edge_vecs = []
            for (i, j) in c1[:5]:
                vec = arr[j] - arr[i]
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    cycle_edge_vecs.append(vec / norm)
            if cycle_edge_vecs:
                mean_cycle = np.mean(cycle_edge_vecs, axis=0)
                norm = np.linalg.norm(mean_cycle)
                if norm > 1e-6:
                    cycle_direction = mean_cycle / norm

        # Void centroid if b2 > 0
        void_centroid = None
        if b2 > 0 and num_c2 > 0:
            triangle_centroids = []
            for (i, j, k) in c2[:5]:
                centroid = (arr[i] + arr[j] + arr[k]) / 3.0
                triangle_centroids.append(centroid)
            if triangle_centroids:
                void_centroid = np.mean(triangle_centroids, axis=0)

        # Cohomological Novelty Score: rewards structural complexity (b1, b2, high cd, sheaf obstruction)
        cohomological_novelty = min(
            1.0,
            float(
                0.25 * (b0 / max(1, N))
                + 0.35 * min(1.0, b1 / 2.0)
                + 0.35 * min(1.0, b2 / 2.0)
                + 0.30 * min(1.0, sheaf_obstruction)
            ),
        )

        return {
            "b0": b0,
            "b1": b1,
            "b2": b2,
            "cohomological_dim": cd,
            "sheaf_obstruction": sheaf_obstruction,
            "cohomological_novelty": cohomological_novelty,
            "cycle_direction": cycle_direction,
            "void_centroid": void_centroid,
        }
