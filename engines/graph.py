class ExpeditionGraph:
    """Topological Latent Graph with Pareto Frontier Traversal."""

    def __init__(self):
        self.nodes = []  # List of dicts
        self.edges = []  # List of dicts

    def add_node(
        self, step, concept, response, novelty, coherence, is_gold, pca_coord
    ):
        node = {
            "id": f"node_{step}",
            "step": step,
            "concept": concept,
            "response": response[:100] + "...",
            "novelty": novelty,
            "coherence": coherence if coherence is not None else 5,
            "is_gold": is_gold,
            "pca_x": pca_coord["x"] if pca_coord else 0,
            "pca_y": pca_coord["y"] if pca_coord else 0,
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

    def get_pareto_frontier(self):
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
        return pareto
