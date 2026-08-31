import re
from collections import Counter
import numpy as np


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
            w_str = str(word).strip()
            if not w_str or len(w_str) < 3:
                continue

            # Tokenize word with and without leading space using llama-cpp-python or HF tokenizer
            variants = [w_str, " " + w_str, w_str.capitalize(), " " + w_str.capitalize()]
            for var in variants:
                try:
                    if hasattr(self.tokenizer, "tokenize"):
                        # llama-cpp-python Llama.tokenize
                        raw_bytes = var.encode("utf-8")
                        toks = self.tokenizer.tokenize(raw_bytes, add_bos=False, special=False)
                    elif hasattr(self.tokenizer, "encode"):
                        toks = self.tokenizer.encode(var, add_special_tokens=False)
                    else:
                        toks = []
                    for t in toks:
                        if isinstance(t, int) and t > 3:  # Skip 0 (empty), 1 (BOS), 2 (EOS), etc.
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


class NoveltyEngine:
    """Cosine Kernel Density Estimation (KDE), Lexical Entropy & Multi-Scale Novelty Evaluator."""

    def __init__(self, bandwidth=0.20, decay_factor=0.97, max_history=150):
        self.bandwidth = bandwidth
        self.decay_factor = decay_factor
        self.max_history = max_history

    def compute_kde_density(self, candidate_vec, history_vecs):
        if not history_vecs:
            return 0.0

        # Bound history array to last max_history vectors to maintain O(1) step time.
        recent_vecs = history_vecs[-self.max_history:]
        hist_arr = np.array(recent_vecs, dtype=np.float64)
        c_vec = np.array(candidate_vec, dtype=np.float64)

        # Normalize vectors to unit sphere for accurate cosine geometry
        norm_c = np.linalg.norm(c_vec) + 1e-9
        c_unit = c_vec / norm_c

        norm_hist = np.linalg.norm(hist_arr, axis=1, keepdims=True) + 1e-9
        hist_unit = hist_arr / norm_hist

        # Cosine similarity s in [-1, 1], cosine distance d in [0, 2]
        cosine_sims = np.dot(hist_unit, c_unit)
        cosine_dists = np.clip(1.0 - cosine_sims, 0.0, 2.0)

        # Calibrated exponential kernel on cosine distance with temporal decay
        K = len(recent_vecs)
        time_weights = np.power(self.decay_factor, K - 1 - np.arange(K))
        kernel_vals = np.exp(-cosine_dists / (self.bandwidth + 1e-6))

        weighted_density = float(np.sum(kernel_vals * time_weights) / (np.sum(time_weights) + 1e-9))
        return min(1.0, max(0.0, weighted_density))

    def compute_lexical_entropy(self, text, history_texts):
        if not history_texts:
            return 1.0

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        if not words:
            return 0.5

        hist_words = re.findall(r"\b[a-zA-Z]{3,}\b", " ".join(history_texts[-15:]).lower())
        hist_counts = Counter(hist_words)
        total_hist = sum(hist_counts.values()) or 1

        # Calculate word novelty ratio: proportion of words that are rare or unseen in history
        rare_or_new = sum(1 for w in words if hist_counts[w] <= 1)
        word_novelty_ratio = rare_or_new / len(words)

        # Information-theoretic surprise (negative log probability)
        surprises = []
        for w in words:
            # Empirical probability with light smoothing
            prob = (hist_counts[w] + 0.1) / (total_hist + 100)
            surprises.append(-np.log2(prob))

        avg_surprise = np.mean(surprises) if surprises else 6.0
        # Calibrate: typical surprise ranges from ~3.5 (highly repetitive) to ~10.0 (completely new)
        norm_surprise = np.clip((avg_surprise - 3.5) / 6.5, 0.0, 1.0)

        # Composite lexical novelty combining surprise and new word ratio
        lexical_novelty = 0.6 * norm_surprise + 0.4 * word_novelty_ratio
        return float(np.clip(lexical_novelty, 0.0, 1.0))

    def evaluate_novelty(
        self, text, candidate_vec, history_vecs, history_texts, cohom_novelty=0.5
    ):
        if not history_vecs:
            return 1.0, candidate_vec

        kde_density = self.compute_kde_density(candidate_vec, history_vecs)
        kde_novelty = max(0.0, 1.0 - kde_density)
        lexical_novelty = self.compute_lexical_entropy(text, history_texts)

        # Balanced Composite Novelty Score fusing Cosine KDE, Lexical Entropy, and Topological Novelty
        composite_novelty = (
            (0.50 * kde_novelty)
            + (0.25 * lexical_novelty)
            + (0.25 * cohom_novelty)
        )
        return float(np.clip(composite_novelty, 0.0, 1.0)), candidate_vec
