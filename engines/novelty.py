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


class NoveltyEngine:
    """Kernel Density Estimation (KDE), Lexical Entropy & Cohomological Novelty Evaluator."""

    def __init__(self, bandwidth=0.35, decay_factor=0.97, max_history=150):
        self.bandwidth = bandwidth
        self.decay_factor = decay_factor
        self.max_history = max_history

    def compute_kde_density(self, candidate_vec, history_vecs):
        if not history_vecs:
            return 0.0

        # Bound history array to last max_history vectors to maintain O(1) step time.
        # Temporal decay weight decay_factor^150 is ~0.01, so older points have negligible weight.
        recent_vecs = history_vecs[-self.max_history:]
        hist_arr = np.array(recent_vecs)
        
        # Compute squared L2 distances
        dists_sq = np.sum((hist_arr - candidate_vec) ** 2, axis=1)

        # Apply Gaussian kernel with exponential temporal decay
        K = len(recent_vecs)
        time_weights = np.power(self.decay_factor, K - 1 - np.arange(K))
        kernel_vals = np.exp(-dists_sq / (2.0 * (self.bandwidth ** 2)))

        weighted_density = np.sum(kernel_vals * time_weights) / np.sum(time_weights)
        return float(weighted_density)

    def compute_lexical_entropy(self, text, history_texts):
        if not history_texts:
            return 1.0

        words = re.findall(r"\w+", text.lower())
        if not words:
            return 0.5

        hist_words = re.findall(r"\w+", " ".join(history_texts[-15:]).lower())
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

    def evaluate_novelty(
        self, text, candidate_vec, history_vecs, history_texts, cohom_novelty=0.5
    ):
        if not history_vecs:
            return 1.0, candidate_vec

        kde_density = self.compute_kde_density(candidate_vec, history_vecs)
        kde_novelty = max(0.0, 1.0 - kde_density)
        lexical_novelty = self.compute_lexical_entropy(text, history_texts)

        # Composite Novelty Score fusing KDE, Lexical Entropy, and Cohomological Dimension
        composite_novelty = (
            (0.50 * kde_novelty)
            + (0.20 * lexical_novelty)
            + (0.30 * cohom_novelty)
        )
        return float(composite_novelty), candidate_vec
