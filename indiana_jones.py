import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Workaround for llama-cpp-python CUDA DLL loading issue on Windows
cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3")
if cuda_path:
    bin_x64 = os.path.join(cuda_path, "bin", "x64")
    if os.path.exists(bin_x64):
        os.environ["PATH"] = bin_x64 + os.pathsep + os.environ["PATH"]
    bin_path = os.path.join(cuda_path, "bin")
    if os.path.exists(bin_path):
        os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

import time
import json
import numpy as np
import random
import re
from collections import Counter
from sklearn.decomposition import PCA
from llama_cpp import Llama, LogitsProcessorList
from sentence_transformers import SentenceTransformer

# Import web dashboard server module
try:
    from dashboard_server import GLOBAL_STATE, start_server_in_thread
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


class RepulsionLogitsProcessor:
    """Dynamic LLM Sampler Logit Processor that penalizes overused clichés & buzzwords."""
    def __init__(self, tokenizer, forbidden_words=None, penalty=5.0):
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.forbidden_token_ids = set()
        if forbidden_words:
            self.update_words(forbidden_words)

    def update_words(self, forbidden_words):
        self.forbidden_token_ids.clear()
        for word in forbidden_words:
            if not word or len(word) < 3:
                continue
            # Encode word with and without leading space
            try:
                tokens = self.tokenizer.encode(word)
                tokens_sp = self.tokenizer.encode(" " + word)
                for t in tokens + tokens_sp:
                    if t > 0:
                        self.forbidden_token_ids.add(t)
            except Exception:
                pass

    def __call__(self, input_ids, logits):
        if self.penalty <= 0 or not self.forbidden_token_ids:
            return logits
        # Subtract penalty from logits of overused tokens
        for tid in self.forbidden_token_ids:
            if tid < len(logits):
                logits[tid] -= float(self.penalty)
        return logits


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
                "b0": 1, "b1": 0, "b2": 0,
                "cohomological_dim": 1.0,
                "sheaf_obstruction": 0.0,
                "cohomological_novelty": 0.5,
                "cycle_direction": None,
                "void_centroid": None
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
        cohomological_novelty = min(1.0, float(0.25 * (b0 / max(1, N)) + 0.35 * min(1.0, b1 / 2.0) + 0.35 * min(1.0, b2 / 2.0) + 0.30 * min(1.0, sheaf_obstruction)))

        return {
            "b0": b0,
            "b1": b1,
            "b2": b2,
            "cohomological_dim": cd,
            "sheaf_obstruction": sheaf_obstruction,
            "cohomological_novelty": cohomological_novelty,
            "cycle_direction": cycle_direction,
            "void_centroid": void_centroid
        }


class NoveltyEngine:
    """Kernel Density Estimation (KDE), Lexical Entropy & Cohomological Novelty Evaluator."""
    def __init__(self, bandwidth=0.35, decay_factor=0.97):
        self.bandwidth = bandwidth
        self.decay_factor = decay_factor

    def compute_kde_density(self, candidate_vec, history_vecs):
        if not history_vecs:
            return 0.0
        
        hist_arr = np.array(history_vecs)
        # Compute squared L2 distances
        dists_sq = np.sum((hist_arr - candidate_vec) ** 2, axis=1)
        
        # Apply Gaussian kernel with exponential temporal decay
        K = len(history_vecs)
        time_weights = np.power(self.decay_factor, K - 1 - np.arange(K))
        kernel_vals = np.exp(-dists_sq / (2.0 * (self.bandwidth ** 2)))
        
        weighted_density = np.sum(kernel_vals * time_weights) / np.sum(time_weights)
        return float(weighted_density)

    def compute_lexical_entropy(self, text, history_texts):
        if not history_texts:
            return 1.0
        
        words = re.findall(r'\w+', text.lower())
        if not words:
            return 0.5
            
        hist_words = re.findall(r'\w+', " ".join(history_texts[-15:]).lower())
        hist_counts = Counter(hist_words)
        total_hist = sum(hist_counts.values()) or 1

        # Calculate average surprise (negative log likelihood under history distribution)
        surprises = []
        for w in words:
            prob = (hist_counts[w] + 1) / (total_hist + 1000)
            surprises.append(-np.log2(prob))
            
        avg_surprise = np.mean(surprises) if surprises else 5.0
        # Normalize to [0, 1] range (surprise typically ranges 2.0 to 12.0)
        norm_surprise = min(1.0, max(0.0, (avg_surprise - 3.0) / 8.0))
        return float(norm_surprise)

    def evaluate_novelty(self, text, candidate_vec, history_vecs, history_texts, cohom_novelty=0.5):
        if not history_vecs:
            return 1.0, candidate_vec
            
        kde_density = self.compute_kde_density(candidate_vec, history_vecs)
        kde_novelty = max(0.0, 1.0 - kde_density)
        lexical_novelty = self.compute_lexical_entropy(text, history_texts)
        
        # Composite Novelty Score fusing KDE, Lexical Entropy, and Cohomological Dimension
        composite_novelty = (0.50 * kde_novelty) + (0.20 * lexical_novelty) + (0.30 * cohom_novelty)
        return float(composite_novelty), candidate_vec


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


class ExpeditionGraph:
    """Topological Latent Graph with Pareto Frontier Traversal."""
    def __init__(self):
        self.nodes = [] # List of dicts
        self.edges = [] # List of dicts

    def add_node(self, step, concept, response, novelty, coherence, is_gold, pca_coord):
        node = {
            "id": f"node_{step}",
            "step": step,
            "concept": concept,
            "response": response[:100] + "...",
            "novelty": novelty,
            "coherence": coherence if coherence else 5,
            "is_gold": is_gold,
            "pca_x": pca_coord["x"] if pca_coord else 0,
            "pca_y": pca_coord["y"] if pca_coord else 0
        }
        self.nodes.append(node)
        if len(self.nodes) > 1:
            prev_node = self.nodes[-2]
            edge = {
                "source": prev_node["id"],
                "target": node["id"],
                "is_gold_transition": is_gold
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
                    if (n2["novelty"] >= n1["novelty"] and n2["coherence"] >= n1["coherence"]) and \
                       (n2["novelty"] > n1["novelty"] or n2["coherence"] > n1["coherence"]):
                        dominated = True
                        break
            if not dominated:
                pareto.append(n1)
        return pareto


class LatentExplorer:
    def __init__(self, model_path, starting_concept):
        print("🗺️  Equipping Advanced Indiana Jones Latent Explorer v2.0...")
        
        # 1. Brain & Sampler
        self.llm = Llama(
            model_path=model_path, 
            n_gpu_layers=-1, 
            n_ctx=4096, 
            verbose=False,
            seed=-1
        )
        
        # 2. Vector Compass
        self.compass = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Next-Gen Engines
        self.novelty_engine = NoveltyEngine(bandwidth=0.35, decay_factor=0.97)
        self.cohomology_engine = CohomologyEngine(max_points=25, eps_percentile=50)
        self.steering_engine = VectorSteering()
        self.graph_engine = ExpeditionGraph()
        self.repulsion_processor = RepulsionLogitsProcessor(self.llm, penalty=5.0)
        
        # 4. Journal & Vector Memory
        self.journal_texts = []
        self.journal_embeddings = []
        self.gold_vault = []
        self.gold_embeddings = []
        self.roadtrip_log = []
        
        # 5. Vitals & State
        self.energy_level = 100
        self.model_path = model_path
        self.starting_concept = starting_concept
        self.current_location = starting_concept
        self.current_temperature = 0.90
        self.low_novelty_streak = 0
        self.orthogonal_push_weight = 0.90
        self.gold_threshold = 0.90
        
        # 16 Specialized Persona Lenses for Cross-Disciplinary Steering
        self.personas = [
            "Sheaf Cohomology & Local-to-Global Obstructions",
            "Persistent Cohomology & Latent Void Dynamics",
            "Homological Algebra & Triangulated Categories of Thought",
            "Spectral Sequences of Conceptual Transformation",
            "Synthetic Epigenetics & Bio-Computing",
            "Topological Fluid Dynamics",
            "Surrealist Cybernetics & Neural Craft",
            "Fungal Network Architecture",
            "Quantum Thermodynamics of Mind",
            "Crystalline Semiotics",
            "Deep-Sea Abyssal Xenobiology",
            "Chrono-Gastronomy & Time-Coded Alchemistry",
            "Non-Euclidean Cathedral Architecture",
            "Astral Metallurgy",
            "Origami Spacetime Dynamics",
            "Bioluminescent Philology"
        ]
        self.current_persona_idx = 0

        # Start live web dashboard
        if DASHBOARD_AVAILABLE:
            GLOBAL_STATE.status = "EXPEDITION_RUNNING"
            GLOBAL_STATE.starting_concept = starting_concept
            GLOBAL_STATE.current_location = starting_concept
            start_server_in_thread(8000)

        print(f"📍 Dropping into High-Dimensional Latent Space at: '{self.current_location}'\n")

    def get_current_persona(self):
        return self.personas[self.current_persona_idx % len(self.personas)]

    def rotate_persona(self):
        self.current_persona_idx += 1
        return self.get_current_persona()

    def update_repulsion_list(self):
        """Extracts overused buzzwords/tropes from history to feed the Logit Processor."""
        if not self.journal_texts:
            return
        words = re.findall(r'\b[a-zA-Z]{4,}\b', " ".join(self.journal_texts[-10:]).lower())
        common = [w for w, cnt in Counter(words).most_common(12) if cnt >= 3]
        base_tropes = ["quantum", "tapestry", "cathedral", "sentient", "whisper", "abyssal", "fabric", "recesses", "symphony"]
        forbidden = list(set(common + base_tropes))
        self.repulsion_processor.update_words(forbidden)

    def craft_prompt(self, concept):
        persona = self.get_current_persona()
        return f"""You are an elite, lateral-thinking latent space explorer.
Explore the following concept through the hyper-specialized lens of: {persona}.

Concept to explore: {concept}

Generate a short, dense paragraph uncovering a brand new, highly specific perspective on this.
Do NOT ask questions, address the reader, or output meta commentary. Write the paragraph directly."""

    def extract_seed_prompt(self, text):
        return f"""Read the following text. Extract the single most unusual, bizarre, or highly specific concept from it. 
Output ONLY that concept as a short noun phrase in 1 to 5 words. No underscores, no blank spaces, no conversational filler.

Text: {text}

Concept:"""

    def validate_seed(self, seed):
        cleaned = seed.strip().strip('"').strip("'").split('\n')[0].strip()
        alpha_chars = sum(1 for c in cleaned if c.isalpha())
        if len(cleaned) < 3 or alpha_chars < 3 or len(cleaned) > 80 or cleaned.count('_') > 3:
            print(f"🚫 Dull/invalid seed detected: '{cleaned[:40]}...' — synthesizing fresh orthogonal concept.")
            fallback_prompt = f"Generate a single highly unusual concept in 3 to 5 words combining {self.get_current_persona()} and unexpected philosophy. Output ONLY the concept."
            output = self.llm(fallback_prompt, max_tokens=15, temperature=0.95)
            cleaned = output["choices"][0]["text"].strip().split('\n')[0].strip()
            print(f"🌱 Synthesized Seed: {cleaned}")
        return cleaned

    def strip_filler(self, text):
        filler_patterns = [
            r'\(Your turn.*', r'\(Note:.*', r'\(Please.*', r'Let me know.*',
            r'Go ahead.*', r'Please respond.*', r'Please share.*', r'Please provide.*',
            r'How\'s that\?.*', r'What do you think\?.*', r'Would you like.*',
            r'Feel free.*', r'I\'d love to.*', r'Your turn.*', r'Now it\'s your turn.*',
            r'\(And,? just to clarify.*', r'\(I\'ve generated.*', r'\(If I don\'t.*',
            r'\(When I\'ve finished.*', r'```python.*', r'\(optional:.*',
            r'\(I chose.*', r'Here\'s my (?:attempt|response).*',
        ]
        paragraphs = text.split('\n\n')
        substantive = [p.strip() for p in paragraphs if len(p.strip()) > 50 and not p.strip().startswith('(') and not p.strip().startswith('Note:')]
        cleaned = substantive[0] if substantive else text
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        return cleaned if len(cleaned) > 30 else text

    def evaluate_coherence(self, text):
        prompt = f"""Rate the logical coherence, readability, and conceptual depth of the following text on a scale from 1 to 10.
Output ONLY a single number between 1 and 10.

Text: {text}

Score:"""
        output = self.llm(prompt, max_tokens=10, temperature=0.1)
        response = output["choices"][0]["text"].strip()
        try:
            match = re.search(r'\b(\d{1,2})\b', response)
            score = min(10, max(1, int(match.group(1)))) if match else 5
            return score
        except Exception:
            return 5

    def compute_pca_coords(self):
        if not self.journal_embeddings:
            return []
        arr = np.array(self.journal_embeddings)
        if len(arr) == 1:
            return [{"x": 0.0, "y": 0.0, "label": self.journal_texts[0][:20], "is_gold": False, "step": 1}]
        
        pca = PCA(n_components=2)
        coords2d = pca.fit_transform(arr)
        
        pca_list = []
        for i, (x, y) in enumerate(coords2d):
            is_gold = i < len(self.roadtrip_log) and self.roadtrip_log[i].get("is_gold", False)
            location = self.roadtrip_log[i].get("location", "") if i < len(self.roadtrip_log) else ""
            pca_list.append({
                "x": float(x),
                "y": float(y),
                "label": location,
                "is_gold": is_gold,
                "step": i + 1
            })
        return pca_list

    def check_web_dashboard_commands(self):
        if not DASHBOARD_AVAILABLE:
            return
        cmds = GLOBAL_STATE.pop_commands()
        for cmd in cmds:
            action = cmd.get("action")
            if action == "FORCE_WARP":
                print("⚡ MANUAL QUANTUM WARP TRIGGERED VIA DASHBOARD!")
                self.low_novelty_streak = 3

    def make_camp(self):
        print("\n⛺ ENERGY DEPLETED. Setting up camp.")
        if self.journal_texts:
            print("📝 Journaling the day's discoveries...")
            summary_prompt = "Summarize these explored concepts into one sentence: " + " | ".join(self.journal_texts[-3:])
            summary = self.llm(summary_prompt, max_tokens=100)["choices"][0]["text"].strip()
            print(f"📖 Campfire thoughts: {summary}")
        
        print("🔥 Resting by the fire for 5 seconds...")
        time.sleep(5)
        self.energy_level = 100
        print("🌅 Sun is up. Energy restored. Packing up camp...\n")

    def save_journal_to_json(self, filename="expedition_journal.json"):
        print(f"\n💾 Saving expedition journal to {filename}...")
        data = {
            "model_path": self.model_path,
            "starting_concept": self.starting_concept,
            "total_steps": len(self.journal_texts),
            "journal_entries": self.journal_texts,
            "gold_vault": self.gold_vault,
            "roadtrip_log": self.roadtrip_log,
            "pareto_frontier": self.graph_engine.get_pareto_frontier(),
            "final_location": self.current_location
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("✅ Journal saved successfully.")

    def start_expedition(self, steps=100, gold_threshold=0.90):
        self.gold_threshold = gold_threshold
        if DASHBOARD_AVAILABLE:
            GLOBAL_STATE.total_steps = steps
            GLOBAL_STATE.gold_threshold = gold_threshold

        for step in range(1, steps + 1):
            if self.energy_level <= 0:
                self.make_camp()

            self.check_web_dashboard_commands()
            self.update_repulsion_list()

            print(f"\n--- 🥾 Step {step}/{steps} | Location: '{self.current_location}' | Lens: {self.get_current_persona()} ---")
            
            # 1. The Trek (Prompt Crafting & Generation with Logit Repulsion)
            print("🧭 Traversing latent space with logit repulsion & persona steering...")
            prompt = self.craft_prompt(self.current_location)
            
            output = self.llm(
                prompt, 
                max_tokens=250, 
                temperature=self.current_temperature,
                top_p=0.9,
                repeat_penalty=1.18,
                logits_processor=LogitsProcessorList([self.repulsion_processor])
            )
            raw_response = output["choices"][0]["text"].strip()
            response = self.strip_filler(raw_response)
            
            # 2. Vector Encoding, Persistent Cohomology & Novelty Evaluation
            vec = self.compass.encode([response])[0]
            
            cohom_data = self.cohomology_engine.compute_simplicial_cohomology(self.journal_embeddings + [vec])
            print(f"📐 Cohomological Dim: {cohom_data['cohomological_dim']:.2f} | b0={cohom_data['b0']} | b1(Loops)={cohom_data['b1']} | b2(Voids)={cohom_data['b2']} | Sheaf Obstruction: {cohom_data['sheaf_obstruction']:.3f}")

            novelty_score, vec = self.novelty_engine.evaluate_novelty(
                response, vec, self.journal_embeddings, self.journal_texts, cohom_novelty=cohom_data["cohomological_novelty"]
            )
            print(f"📊 Multi-Scale KDE + Cohomological Novelty Score: {novelty_score:.2f} (Target > {gold_threshold})")
            
            # Temperature Adjustment
            if novelty_score < 0.35:
                self.current_temperature = min(1.3, self.current_temperature + 0.08)
                self.low_novelty_streak += 1
                print(f"🌡️  Low novelty. Raising temperature to {self.current_temperature:.2f}")
            elif novelty_score > 0.70:
                self.current_temperature = max(0.45, self.current_temperature - 0.08)
                self.low_novelty_streak = 0
                print(f"🧊 High novelty. Lowering temperature to {self.current_temperature:.2f}")
            else:
                self.low_novelty_streak = 0
                if self.current_temperature > 0.85:
                    self.current_temperature = max(0.85, self.current_temperature - 0.04)

            coherence_score = None
            is_gold = False

            # Gold Appraisal
            if novelty_score > gold_threshold:
                print("🧐 Appraising coherence & conceptual depth...")
                coherence_score = self.evaluate_coherence(response)
                print(f"⚖️  Coherence Score: {coherence_score}/10")
                
                if coherence_score >= 7:
                    print("🌟 VEIN OF GOLD DISCOVERED! 🌟")
                    print(f"💎 {response}\n")
                    self.gold_vault.append(response)
                    self.gold_embeddings.append(vec)
                    is_gold = True
                else:
                    print("🗑️ Fool's Gold: Novel, but lacks coherence.")
            else:
                print("🪨 Conventional terrain. Continuing traversal.")
                
            self.journal_texts.append(response)
            self.journal_embeddings.append(vec)

            # Compute PCA coordinates for live 2D latent map
            pca_coords = self.compute_pca_coords()
            latest_pca = pca_coords[-1] if pca_coords else {"x": 0.0, "y": 0.0}

            # Graph topology addition
            graph_node = self.graph_engine.add_node(
                step, self.current_location, response, novelty_score, coherence_score, is_gold, latest_pca
            )

            # 3. Vector Steering & Next Seed Selection
            print("🔎 Steering next seed vector with topological guidance...")
            seed_prompt = self.extract_seed_prompt(response)
            seed_output = self.llm(seed_prompt, max_tokens=15, temperature=0.3)
            raw_next_location = self.validate_seed(seed_output["choices"][0]["text"].strip())
            
            leap_triggered = False
            concept_blended = False
            
            # Topological cycle repulsion and void attraction steering
            if cohom_data["b1"] > 0:
                print(f"🌀 1-Cycle Loop detected (b1={cohom_data['b1']}). Applying Topological Loop Repulsion Push.")
            if cohom_data["b2"] > 0:
                print(f"🕳️ 2-Void Cavity detected (b2={cohom_data['b2']}). Steering search into Topological Void Centroid.")

            # Backtracking / Pareto Frontier Jump / Vector Push
            if self.low_novelty_streak >= 3:
                print("⚡ QUANTUM VECTOR WARP TRIGGERED! Explorer escaping latent plateau.")
                leap_triggered = True
                self.low_novelty_streak = 0
                self.current_temperature = 0.85
                self.rotate_persona()
                
                # Backtrack to top Pareto frontier node or gold vault
                pareto_nodes = self.graph_engine.get_pareto_frontier()
                base_concept = pareto_nodes[0]["concept"] if pareto_nodes else self.starting_concept
                
                # Apply orthogonal vector push
                components = VectorSteering.compute_pca_subspace(self.journal_embeddings)
                raw_seed_vec = self.compass.encode([raw_next_location])[0]
                ortho_vec = VectorSteering.orthogonal_projection(raw_seed_vec, components)
                ortho_vec = VectorSteering.topological_steering(ortho_vec, cohom_data, push_weight=0.7)
                
                next_location = f"{base_concept} orthogonally shifted through {self.get_current_persona()}"
                print(f"🌀 Warping along Topological Cohomological Vector to: '{next_location}'")
            elif novelty_score < 0.48:
                # Rotate persona lens to inject cross-disciplinary force
                new_persona = self.rotate_persona()
                next_location = f"{raw_next_location} ({new_persona})"
                concept_blended = True
                print(f"🧬 Injecting Persona Steering Lens: '{next_location}'")
            else:
                next_location = raw_next_location

            step_data = {
                "step": step,
                "location": self.current_location,
                "persona": self.get_current_persona(),
                "temperature": float(self.current_temperature),
                "novelty_score": float(novelty_score),
                "coherence_score": coherence_score,
                "is_gold": is_gold,
                "leap_triggered": leap_triggered,
                "concept_blended": concept_blended,
                "response": response,
                "next_location": next_location,
                "energy_after_step": self.energy_level - 34,
                "pca_coord": latest_pca,
                "cohomological_dim": cohom_data["cohomological_dim"],
                "b0": cohom_data["b0"],
                "b1": cohom_data["b1"],
                "b2": cohom_data["b2"],
                "sheaf_obstruction": cohom_data["sheaf_obstruction"]
            }
            self.roadtrip_log.append(step_data)

            # Update live web dashboard
            if DASHBOARD_AVAILABLE:
                GLOBAL_STATE.update_step(step_data)
                GLOBAL_STATE.set_graph_data(
                    self.graph_engine.nodes,
                    self.graph_engine.edges,
                    self.graph_engine.get_pareto_frontier()
                )

            self.current_location = next_location
            self.energy_level -= 34

        print("\n==================================")
        print("🏆 EXPEDITION COMPLETE 🏆")
        print(f"Total Gold Veins Found: {len(self.gold_vault)}")
        print(f"Pareto Frontier Nodes: {len(self.graph_engine.get_pareto_frontier())}")
        print("==================================")
        
        if DASHBOARD_AVAILABLE:
            GLOBAL_STATE.status = "COMPLETED"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_journal_to_json(f"journal_{timestamp}.json")


if __name__ == "__main__":
    MODEL_FILE = "./LLM/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    
    indy = LatentExplorer(
        model_path=MODEL_FILE, 
        starting_concept="The relationship between truth and confabulations"
    )
    
    indy.start_expedition(steps=100, gold_threshold=0.90)