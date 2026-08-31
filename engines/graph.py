class ExpeditionGraph:
    """Topological Latent Graph with Pareto Frontier Traversal."""

    def __init__(self):
        self.nodes = []  # List of dicts
        self.edges = []  # List of dicts

    def add_node(
        self, step, concept, response, novelty, coherence, is_gold, pca_coord, cohomological_dim=1.0
    ):
        coh_val = float(coherence) if coherence is not None else 5.0
        cd_val = float(cohomological_dim) if cohomological_dim is not None else 1.0
        node = {
            "id": f"node_{step}",
            "step": step,
            "concept": concept,
            "response": response[:120] + "..." if len(response) > 120 else response,
            "novelty": float(novelty),
            "coherence": coh_val,
            "cohomological_dim": cd_val,
            "is_gold": bool(is_gold),
            "pca_x": float(pca_coord["x"]) if pca_coord and "x" in pca_coord else 0.0,
            "pca_y": float(pca_coord["y"]) if pca_coord and "y" in pca_coord else 0.0,
            "pca_z": float(pca_coord["z"]) if pca_coord and "z" in pca_coord else 0.0,
        }
        self.nodes.append(node)
        if len(self.nodes) > 1:
            prev_node = self.nodes[-2]
            edge = {
                "source": prev_node["id"],
                "target": node["id"],
                "is_gold_transition": is_gold,
            }
            self.edges.append(edge)
        return node

    def update_pca_coords(self, aligned_pca_coords):
        """Realigns all nodes with the latest global PCA coordinate system and topological metrics."""
        for i, coord in enumerate(aligned_pca_coords):
            if i < len(self.nodes):
                self.nodes[i]["pca_x"] = float(coord.get("x", 0.0))
                self.nodes[i]["pca_y"] = float(coord.get("y", 0.0))
                self.nodes[i]["pca_z"] = float(coord.get("z", 0.0))
                if "novelty" in coord:
                    self.nodes[i]["novelty"] = float(coord["novelty"])
                if "coherence" in coord:
                    self.nodes[i]["coherence"] = float(coord["coherence"])
                if "cohomological_dim" in coord:
                    self.nodes[i]["cohomological_dim"] = float(coord["cohomological_dim"])
                if "is_gold" in coord:
                    self.nodes[i]["is_gold"] = bool(coord["is_gold"])

    def get_pareto_frontier(self):
        """Extracts non-dominated nodes across Novelty and Coherence."""
        if not self.nodes:
            return []

        pareto = []
        for i, n1 in enumerate(self.nodes):
            dominated = False
            for j, n2 in enumerate(self.nodes):
                if i != j:
                    if (
                        n2["novelty"] >= n1["novelty"]
                        and n2["coherence"] >= n1["coherence"]
                    ) and (
                        n2["novelty"] > n1["novelty"]
                        or n2["coherence"] > n1["coherence"]
                    ):
                        dominated = True
                        break
            if not dominated:
                pareto.append(n1)
        # Sort by composite product of novelty and coherence descending
        pareto.sort(key=lambda n: n["novelty"] * (n["coherence"] / 10.0), reverse=True)
        return pareto
