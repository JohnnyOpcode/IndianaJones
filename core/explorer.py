import os
import sys

# Setup Windows CUDA DLL search paths before importing llama_cpp
import core.dll_setup

import time
import json
import numpy as np
import random
import re
from collections import Counter
from sklearn.decomposition import PCA

# Import torch and SentenceTransformer first to initialize PyTorch/OpenMP runtime before llama_cpp
import torch
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama, LogitsProcessorList

from engines.cohomology import CohomologyEngine
from engines.novelty import NoveltyEngine, RepulsionLogitsProcessor
from engines.steering import VectorSteering
from engines.graph import ExpeditionGraph
from core.config import ExplorerConfig

# Singleton cached SentenceTransformer model to prevent reload deadlocks
_COMPASS_MODEL = None

def get_compass_model():
    global _COMPASS_MODEL
    if _COMPASS_MODEL is None:
        print("🧭 Loading SentenceTransformer vector compass (all-MiniLM-L6-v2)...")
        try:
            _COMPASS_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu', local_files_only=True)
        except Exception:
            _COMPASS_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return _COMPASS_MODEL

# Pre-warm compass model on CPU before llama_cpp can touch CUDA runtime
get_compass_model()

# Import web dashboard server module
try:
    from dashboard_server import GLOBAL_STATE, start_server_in_thread
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


class LatentExplorer:
    def __init__(self, config: ExplorerConfig):
        print("🗺️  Equipping Advanced Indiana Jones Latent Explorer v2.2", flush=True)
        self.config = config
        
        # 1. Vector Compass
        self.compass = get_compass_model()

        # 2. Brain & Sampler
        print(f"🧠 Loading Llama LLM model from: {config.model_path} (GPU layers: all)...", flush=True)
        self.llm = Llama(
            model_path=config.model_path, 
            n_gpu_layers=-1, 
            n_ctx=4096, 
            verbose=False,
            seed=-1
        )
        print("   -> Llama LLM neural engine ready.", flush=True)
        
        # 3. Next-Gen Engines
        self.novelty_engine = NoveltyEngine(bandwidth=0.20, decay_factor=0.97, max_history=150)
        self.cohomology_engine = CohomologyEngine(max_points=30, eps_percentiles=(25, 50, 75))
        self.steering_engine = VectorSteering()
        self.graph_engine = ExpeditionGraph()
        self.repulsion_processor = RepulsionLogitsProcessor(self.llm, penalty=5.0)
        
        # 4. Journal & Vector Memory
        self.journal_texts = []
        self.journal_embeddings = []
        self.gold_vault = []
        self.gold_embeddings = []
        self.roadtrip_log = []
        
        # 5. Vitals, Anchors & State
        self.energy_level = 100
        self.model_path = config.model_path
        self.starting_concept = config.starting_concept
        self.current_location = config.starting_concept
        self.starting_vec = self.compass.encode([config.starting_concept])[0]
        print("   -> Starting concept encoded into 384-D latent vector.", flush=True)
        self.current_temperature = config.temperature
        self.low_novelty_streak = 0
        self.orthogonal_push_weight = 0.90
        self.gold_threshold = config.gold_threshold
        
        # 16 Balanced Multidisciplinary Persona Lenses
        self.personas = [
            "Mechanistic Systems Biology & Network Medicine",
            "Nonlinear Dynamics & Bifurcation Theory",
            "Information Theory & Algorithmic Complexity",
            "Persistent Homology & Latent Topology",
            "Synthetic Epigenetics & Bio-Computing",
            "Thermodynamics of Complex Adaptive Systems",
            "Surrealist Cybernetics & Cognitive Architecture",
            "Fungal & Mycelial Network Architectures",
            "Quantum Information & Coherence Dynamics",
            "Crystalline Semiotics & Structural Linguistics",
            "Deep-Sea Abyssal Xenobiology",
            "Evolutionary Game Theory & Niche Construction",
            "Non-Euclidean Spatial Morphologies",
            "Material Science & Metamaterial Engineering",
            "Category Theory & Conceptual Translation",
            "Biomimetic Energy Dynamics"
        ]
        self.current_persona_idx = 0
        self.restart_requested = False

        # Start live web dashboard
        if DASHBOARD_AVAILABLE and config.enable_dashboard:
            with GLOBAL_STATE.lock:
                GLOBAL_STATE.status = "EXPEDITION_RUNNING"
                GLOBAL_STATE.starting_concept = config.starting_concept
                GLOBAL_STATE.current_location = config.starting_concept
                GLOBAL_STATE.set_personas(self.personas)
            start_server_in_thread(config.port)

        print(f"📍 Dropping into High-Dimensional Latent Space at: '{self.current_location}'\n", flush=True)

    def reset_expedition_state(self):
        print("\n🔄 RESTARTING EXPEDITION FROM STARTING CONCEPT...")
        self.journal_texts.clear()
        self.journal_embeddings.clear()
        self.gold_vault.clear()
        self.gold_embeddings.clear()
        self.roadtrip_log.clear()
        self.energy_level = 100
        self.current_location = self.starting_concept
        self.starting_vec = self.compass.encode([self.starting_concept])[0]
        self.low_novelty_streak = 0
        self.current_persona_idx = 0
        self.graph_engine = ExpeditionGraph()
        self.novelty_engine = NoveltyEngine(bandwidth=0.20, decay_factor=0.97, max_history=150)
        self.cohomology_engine = CohomologyEngine(max_points=30, eps_percentiles=(25, 50, 75))
        self.restart_requested = True
        if DASHBOARD_AVAILABLE and self.config.enable_dashboard:
            GLOBAL_STATE.reset_expedition()

    def get_current_persona(self):
        return self.personas[self.current_persona_idx % len(self.personas)]

    def rotate_persona(self):
        self.current_persona_idx = (self.current_persona_idx + 1) % len(self.personas)
        return self.get_current_persona()

    def update_repulsion_list(self):
        """Extracts overused buzzwords/tropes from recent history to feed the Logit Processor."""
        if not self.journal_texts:
            return
        words = re.findall(r'\b[a-zA-Z]{4,}\b', " ".join(self.journal_texts[-10:]).lower())
        common = [w for w, cnt in Counter(words).most_common(15) if cnt >= 3]
        base_tropes = [
            "tapestry", "cathedral", "sentient", "whisper", "fabric",
            "recesses", "symphony", "dance", "realm", "interplay",
            "intricate", "woven", "nexus", "echoes", "portal"
        ]
        forbidden = list(set(common + base_tropes))
        self.repulsion_processor.update_words(forbidden)

    def generate_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, 
                      temperature: float = 0.85, top_p: float = 0.9, repeat_penalty: float = 1.18, 
                      logits_processor=None, stream_to_console: bool = True) -> str:
        """Robust chat completion generator with real-time token streaming and Llama 3 chat template."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 1. Try create_chat_completion with live token streaming
        try:
            kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "stream": stream_to_console
            }
            if logits_processor is not None:
                kwargs["logits_processor"] = logits_processor
                
            if stream_to_console:
                stream_resp = self.llm.create_chat_completion(**kwargs)
                collected = []
                for chunk in stream_resp:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        collected.append(content)
                        sys.stdout.write(content)
                        sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()
                res_str = "".join(collected).strip()
                if res_str:
                    return res_str
            else:
                response = self.llm.create_chat_completion(**kwargs)
                content = response["choices"][0]["message"].get("content", "")
                if content and content.strip():
                    return content.strip()
        except Exception as e:
            print(f"⚠️ Live chat streaming note: {e}. Falling back to standard text completion...", flush=True)

        # 2. Fallback to raw text completion
        try:
            prompt_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            output = self.llm(
                prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                logits_processor=logits_processor
            )
            raw = output["choices"][0]["text"].strip()
            print(raw, flush=True)
            return raw
        except Exception as e2:
            print(f"❌ Generation error: {e2}. Returning fallback text.", flush=True)
            return "An abstract nexus of latent concepts awaiting further topological exploration."

    def craft_prompt(self, current_location, contrast_cues=None):
        """Preserves the starting domain anchor while steering into novel orthogonal perspectives."""
        persona = self.get_current_persona()
        system_prompt = (
            f"You are an elite, lateral-thinking researcher investigating: '{self.starting_concept}'. "
            f"Your current analytical lens is: '{persona}'. "
            f"Uncover rigorous, highly original, non-obvious cross-disciplinary insights."
        )
        
        contrast_str = ""
        if contrast_cues:
            contrast_str = f" Orthogonally depart from well-trodden themes ({', '.join(contrast_cues)})."

        user_prompt = (
            f"Primary Domain: {self.starting_concept}\n"
            f"Current Exploration Front: {current_location}.{contrast_str}\n\n"
            f"Generate a dense, specific, insightful paragraph connecting this front to the primary domain. "
            f"Avoid conversational filler, questions, and cliché metaphors. Write the insight directly."
        )
        return system_prompt, user_prompt

    def extract_seed_prompt(self, text):
        system_prompt = "You are a precise concept extraction engine."
        user_prompt = (
            f"Read the following text. Extract the single most novel, highly specific mechanism or concept from it "
            f"that relates to '{self.starting_concept}'.\n"
            f"Output ONLY that concept as a short noun phrase in 1 to 5 words. No formatting, no conversational filler.\n\n"
            f"Text: {text}\n\nConcept:"
        )
        return system_prompt, user_prompt

    def validate_seed(self, seed):
        cleaned = seed.strip().strip('"').strip("'").split('\n')[0].strip()
        cleaned = re.sub(r'^(concept:|the\s+concept\s+of:?)\s*', '', cleaned, flags=re.IGNORECASE)
        alpha_chars = sum(1 for c in cleaned if c.isalpha())
        if len(cleaned) < 3 or alpha_chars < 3 or len(cleaned) > 80 or cleaned.count('_') > 3:
            print(f"🚫 Dull/invalid seed detected: '{cleaned[:40]}...' — synthesizing fresh orthogonal concept.")
            sys_p = "You are a scientific concept synthesizer."
            usr_p = f"Generate a single highly novel concept in 3 to 5 words bridging '{self.starting_concept}' and '{self.get_current_persona()}'. Output ONLY the concept."
            cleaned = self.generate_chat(sys_p, usr_p, max_tokens=20, temperature=0.9).strip().split('\n')[0].strip()
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

    def compute_coherence(self, text, vec, novelty_score):
        """Computes continuous semantic coherence & depth score [1.0, 10.0] for all steps."""
        # 1. Semantic alignment to starting domain (Cosine similarity mapped to 1-10)
        norm_v = np.linalg.norm(vec) + 1e-9
        norm_start = np.linalg.norm(self.starting_vec) + 1e-9
        cos_sim = float(np.dot(vec / norm_v, self.starting_vec / norm_start))
        domain_coherence = float(np.clip(1.0 + 9.0 * max(0.0, cos_sim), 1.0, 10.0))

        # 2. Structural substance check (word count & density)
        word_count = len(re.findall(r'\b\w+\b', text))
        substance_factor = min(1.0, max(0.4, word_count / 50.0))

        base_coherence = float(domain_coherence * substance_factor)

        # 3. Targeted LLM evaluation for candidate gold discoveries
        if novelty_score > self.gold_threshold:
            sys_p = "You are a rigorous, critical scientific reviewer."
            usr_p = (
                f"Assess the logical rigor, depth, and substantive relevance of the following insight regarding '{self.starting_concept}'. "
                f"Rate strictly from 1 to 10 (where 10 is groundbreaking and rigorous, 1 is vague gibberish).\n"
                f"Output ONLY a single integer between 1 and 10.\n\nText: {text}\n\nScore:"
            )
            response = self.generate_chat(sys_p, usr_p, max_tokens=10, temperature=0.1)
            try:
                match = re.search(r'\b(\d{1,2})\b', response)
                llm_score = float(min(10, max(1, int(match.group(1))))) if match else 6.0
                return round(0.4 * base_coherence + 0.6 * llm_score, 1)
            except Exception:
                pass
        return round(base_coherence, 1)

    def compute_pca_coords(self, current_step_meta=None):
        """Re-transforms all historical embeddings into a consistent 2D/3D visual topological projection."""
        if not self.journal_embeddings:
            return []
        arr = np.array(self.journal_embeddings, dtype=np.float64)
        if len(arr) == 1:
            log_entry = current_step_meta if current_step_meta else (self.roadtrip_log[0] if self.roadtrip_log else {})
            nov = float(log_entry.get("novelty_score", 1.0))
            coh = float(log_entry.get("coherence_score", 5.0))
            cd = float(log_entry.get("cohomological_dim", 1.0))
            is_g = bool(log_entry.get("is_gold", False))
            lbl = log_entry.get("location", self.journal_texts[0][:24])
            return [{
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "label": lbl,
                "is_gold": is_g,
                "step": 1,
                "novelty": nov,
                "coherence": coh,
                "cohomological_dim": cd
            }]
        
        # Fit PCA with up to 3 components to support both 2D trajectories and 3D visual topology surfaces
        n_comps = min(3, len(arr))
        pca = PCA(n_components=n_comps)
        if len(arr) > 200:
            pca.fit(arr[-200:])
        else:
            pca.fit(arr)
        coords = pca.transform(arr)
        
        pca_list = []
        for i, pt in enumerate(coords):
            x = float(pt[0])
            y = float(pt[1]) if n_comps >= 2 else 0.0
            z = float(pt[2]) if n_comps >= 3 else 0.0
            if i < len(self.roadtrip_log):
                log_entry = self.roadtrip_log[i]
            elif current_step_meta and i == len(self.roadtrip_log):
                log_entry = current_step_meta
            else:
                log_entry = {}
            is_gold = bool(log_entry.get("is_gold", False))
            location = log_entry.get("location", "") if log_entry else (self.journal_texts[i][:24] if i < len(self.journal_texts) else "")
            novelty = float(log_entry.get("novelty_score", 0.0))
            coherence = float(log_entry.get("coherence_score", 5.0))
            cohom_dim = float(log_entry.get("cohomological_dim", 1.0))
            pca_list.append({
                "x": x,
                "y": y,
                "z": z,
                "label": location,
                "is_gold": is_gold,
                "step": i + 1,
                "novelty": novelty,
                "coherence": coherence,
                "cohomological_dim": cohom_dim
            })
        
        # Re-align graph engine node coordinates in bulk
        self.graph_engine.update_pca_coords(pca_list)
        return pca_list

    def check_web_dashboard_commands(self):
        if not DASHBOARD_AVAILABLE or not self.config.enable_dashboard:
            return
        
        # Process any pending commands
        cmds = GLOBAL_STATE.pop_commands()
        for cmd in cmds:
            action = cmd.get("action")
            if action == "FORCE_WARP":
                print("⚡ MANUAL QUANTUM WARP TRIGGERED VIA DASHBOARD!")
                self.low_novelty_streak = 3
            elif action == "RESTART":
                self.reset_expedition_state()
            elif action == "TELEPORT_CONCEPT":
                target = cmd.get("target_concept", "").strip()
                if target:
                    print(f"🚀 CONCEPT TELEPORTATION VIA DASHBOARD → '{target}'")
                    self.current_location = target
                    self.low_novelty_streak = 0
            elif action == "SET_PERSONA":
                idx = int(cmd.get("persona_idx", self.current_persona_idx))
                if 0 <= idx < len(self.personas):
                    self.current_persona_idx = idx
                    print(f"🔭 Persona Lens switched to: '{self.personas[idx]}'")
            elif action == "UPDATE_PARAMS":
                if "gold_threshold" in cmd:
                    self.gold_threshold = float(cmd["gold_threshold"])
                    print(f"🎛️ Dynamic Gold Threshold updated to {self.gold_threshold:.2f}")
                if "orthogonal_push_weight" in cmd:
                    self.orthogonal_push_weight = float(cmd["orthogonal_push_weight"])
                    print(f"🎛️ Dynamic Push Weight updated to {self.orthogonal_push_weight:.2f}")
                if "repulsion_strength" in cmd:
                    self.repulsion_processor.penalty = float(cmd["repulsion_strength"])
                    print(f"🎛️ Dynamic Repulsion Penalty updated to {self.repulsion_processor.penalty:.1f}")
                if "temperature" in cmd:
                    self.current_temperature = float(cmd["temperature"])
                    print(f"🎛️ Dynamic Temperature updated to {self.current_temperature:.2f}")

        # Always sync current runtime state back to GLOBAL_STATE
        with GLOBAL_STATE.lock:
            self.gold_threshold = GLOBAL_STATE.gold_threshold
            self.orthogonal_push_weight = GLOBAL_STATE.orthogonal_push_weight
            self.repulsion_processor.penalty = GLOBAL_STATE.repulsion_strength

        # Handle pause state cleanly
        while GLOBAL_STATE.status == "PAUSED":
            time.sleep(0.5)
            pause_cmds = GLOBAL_STATE.pop_commands()
            for cmd in pause_cmds:
                action = cmd.get("action")
                if action == "RESUME":
                    print("▶️ Expedition resumed via dashboard.")
                    break
                elif action == "RESTART":
                    self.reset_expedition_state()
                    break
                elif action == "UPDATE_PARAMS":
                    if "gold_threshold" in cmd:
                        self.gold_threshold = float(cmd["gold_threshold"])
                    if "orthogonal_push_weight" in cmd:
                        self.orthogonal_push_weight = float(cmd["orthogonal_push_weight"])
                    if "repulsion_strength" in cmd:
                        self.repulsion_processor.penalty = float(cmd["repulsion_strength"])
                    if "temperature" in cmd:
                        self.current_temperature = float(cmd["temperature"])

    def make_camp(self):
        """Synthesizes recent exploration milestones without artificial blocking sleep."""
        print("\n⛺ REACHED EXPLORATION CAMP. Consolidating field observations...")
        if self.journal_texts:
            sys_p = "You are a reflective lead scientist."
            usr_p = f"Summarize key breakthroughs regarding '{self.starting_concept}' from these observations in one sentence: " + " | ".join(self.journal_texts[-3:])
            summary = self.generate_chat(sys_p, usr_p, max_tokens=100, temperature=0.5)
            print(f"📖 Field Synthesis: {summary}")
        
        self.energy_level = 100
        print("🌅 Energy replenished. Traversal resumes immediately.\n")

    def save_journal_to_json(self, filename=None):
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"journal_{timestamp}.json"
        
        filepath = os.path.join(self.config.output_dir, filename)
        print(f"\n💾 Saving expedition journal to {filepath}...")
        
        data = {
            "model_path": self.model_path,
            "starting_concept": self.starting_concept,
            "total_steps": len(self.journal_texts),
            "journal_entries": self.journal_texts,
            "journal_history": self.roadtrip_log,
            "gold_vault": self.gold_vault,
            "roadtrip_log": self.roadtrip_log,
            "pareto_frontier": self.graph_engine.get_pareto_frontier(),
            "pca_coords": self.compute_pca_coords(),
            "final_location": self.current_location
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("✅ Journal saved successfully.")

    def start_expedition(self, steps=None, gold_threshold=None):
        if steps is None:
            steps = self.config.steps
        if gold_threshold is None:
            gold_threshold = self.config.gold_threshold
            
        self.gold_threshold = gold_threshold
        if DASHBOARD_AVAILABLE and self.config.enable_dashboard:
            with GLOBAL_STATE.lock:
                GLOBAL_STATE.total_steps = steps
                GLOBAL_STATE.gold_threshold = gold_threshold

        contrast_cues = []
        step = 1
        while step <= steps:
            if self.restart_requested:
                self.restart_requested = False
                step = 1
                contrast_cues = []

            if self.energy_level <= 0:
                self.make_camp()

            self.check_web_dashboard_commands()
            if self.restart_requested:
                self.restart_requested = False
                step = 1
                contrast_cues = []
                continue

            self.update_repulsion_list()

            print(f"\n--- 🥾 Step {step}/{steps} | Location: '{self.current_location}' | Lens: {self.get_current_persona()} ---", flush=True)
            
            # 1. The Trek (Prompt Crafting & Generation with Logit Repulsion & Steered Contrast Cues)
            print("🧭 Traversing latent space with repulsion logits & orthogonal guidance...", flush=True)
            sys_prompt, user_prompt = self.craft_prompt(self.current_location, contrast_cues=contrast_cues)
            
            raw_response = self.generate_chat(
                sys_prompt,
                user_prompt,
                max_tokens=250,
                temperature=self.current_temperature,
                top_p=0.9,
                repeat_penalty=1.18,
                logits_processor=LogitsProcessorList([self.repulsion_processor]),
                stream_to_console=True
            )
            response = self.strip_filler(raw_response)
            
            # 2. Vector Encoding, Simplicial Topology & Novelty Evaluation
            vec = self.compass.encode([response])[0]
            
            cohom_data = self.cohomology_engine.compute_simplicial_cohomology(self.journal_embeddings + [vec])
            print(
                f"📐 Topological Dim: {cohom_data['cohomological_dim']:.2f} | "
                f"b0={cohom_data['b0']} | b1(Loops)={cohom_data['b1']} | b2(Voids)={cohom_data['b2']} | "
                f"Sheaf Obstruction: {cohom_data['sheaf_obstruction']:.3f} | Persistence Entropy: {cohom_data['persistence_entropy']:.2f}",
                flush=True
            )

            novelty_score, vec = self.novelty_engine.evaluate_novelty(
                response, vec, self.journal_embeddings, self.journal_texts, cohom_novelty=cohom_data["cohomological_novelty"]
            )
            print(f"📊 Cosine KDE + Lexical + Topological Novelty: {novelty_score:.2f} (Target > {self.gold_threshold:.2f})", flush=True)
            
            # Dynamic Temperature & Streak Tracking
            if novelty_score < 0.35:
                self.current_temperature = min(1.25, self.current_temperature + 0.08)
                self.low_novelty_streak += 1
                print(f"🌡️  Low novelty detected. Raising temperature to {self.current_temperature:.2f} (Streak: {self.low_novelty_streak}/3)", flush=True)
            elif novelty_score > 0.65:
                self.current_temperature = max(0.65, self.current_temperature - 0.05)
                self.low_novelty_streak = 0
                print(f"🧊 High novelty confirmed. Stabilizing temperature to {self.current_temperature:.2f}", flush=True)
            else:
                self.low_novelty_streak = 0
                if self.current_temperature > 0.85:
                    self.current_temperature = max(0.85, self.current_temperature - 0.03)

            # Continuous Coherence Appraisal across all steps
            coherence_score = self.compute_coherence(response, vec, novelty_score)
            is_gold = False

            if novelty_score >= self.gold_threshold and coherence_score >= 6.5:
                print("🌟 VEIN OF GOLD DISCOVERED! (Registered to Gold Vault) 🌟", flush=True)
                self.gold_vault.append(response)
                self.gold_embeddings.append(vec)
                is_gold = True
            elif novelty_score >= self.gold_threshold:
                print(f"🗑️ High novelty ({novelty_score:.2f}) but insufficient coherence ({coherence_score}/10).", flush=True)
            else:
                print(f"🪨 Traversed terrain. Coherence: {coherence_score}/10.", flush=True)
                
            self.journal_texts.append(response)
            self.journal_embeddings.append(vec)

            # Re-transform and align 2D/3D PCA coordinates across all history
            current_meta = {
                "location": self.current_location,
                "is_gold": is_gold,
                "novelty_score": float(novelty_score),
                "coherence_score": float(coherence_score),
                "cohomological_dim": float(cohom_data["cohomological_dim"])
            }
            pca_coords = self.compute_pca_coords(current_step_meta=current_meta)
            latest_pca = pca_coords[-1] if pca_coords else {
                "x": 0.0, "y": 0.0, "z": 0.0,
                "novelty": novelty_score, "coherence": coherence_score,
                "cohomological_dim": cohom_data["cohomological_dim"],
                "is_gold": is_gold, "label": self.current_location, "step": step
            }

            # Graph topology addition
            graph_node = self.graph_engine.add_node(
                step, self.current_location, response, novelty_score, coherence_score, is_gold, latest_pca,
                cohomological_dim=cohom_data["cohomological_dim"]
            )

            # 3. Vector Steering & Next Seed Derivation
            print("🔎 Steering next seed vector with subspace projection & topological cues...", flush=True)
            seed_sys, seed_usr = self.extract_seed_prompt(response)
            seed_output = self.generate_chat(seed_sys, seed_usr, max_tokens=15, temperature=0.3, stream_to_console=False)
            raw_next_location = self.validate_seed(seed_output)
            print(f"👉 Next Latent Concept Seed: '{raw_next_location}'", flush=True)
            
            # Calculate authentic high-dimensional steered vector
            raw_seed_vec = self.compass.encode([raw_next_location])[0]
            components = VectorSteering.compute_pca_subspace(self.journal_embeddings, k=3)
            ortho_vec = VectorSteering.orthogonal_projection(raw_seed_vec, components)
            ortho_vec = VectorSteering.topological_steering(ortho_vec, cohom_data, push_weight=self.orthogonal_push_weight)
            if self.gold_embeddings:
                ortho_vec = VectorSteering.apply_momentum_steering(ortho_vec, self.gold_embeddings, momentum_weight=0.25)

            # Extract contrast themes to bridge steered vector into prompt space
            contrast_cues = VectorSteering.derive_semantic_contrast_themes(
                ortho_vec, self.journal_embeddings, [s.get("location", "") for s in self.roadtrip_log], top_k=2
            )

            leap_triggered = False
            concept_blended = False
            
            # Topological cycle repulsion and void attraction logging
            if cohom_data["b1"] > 0:
                print(f"🌀 1-Cycle Loop detected (b1={cohom_data['b1']}). Applying Harmonic Loop Repulsion.", flush=True)
            if cohom_data["b2"] > 0:
                print(f"🕳️ 2-Void Cavity detected (b2={cohom_data['b2']}). Steering search into Topological Void Centroid.", flush=True)

            # Backtracking / Pareto Frontier Jump / Vector Push
            if self.low_novelty_streak >= 3:
                print("⚡ QUANTUM VECTOR WARP TRIGGERED! Escaping latent plateau.", flush=True)
                leap_triggered = True
                self.low_novelty_streak = 0
                self.current_temperature = 0.90
                self.rotate_persona()
                
                # Backtrack to non-dominated Pareto frontier node
                pareto_nodes = self.graph_engine.get_pareto_frontier()
                base_concept = pareto_nodes[0]["concept"] if pareto_nodes else self.starting_concept
                
                next_location = f"{base_concept} (cross-mapped through {self.get_current_persona()})"
                print(f"🌀 Warping along Steered Pareto Frontier Vector to: '{next_location}'", flush=True)
            elif novelty_score < 0.45 or step % 3 == 0:
                # Rotate persona lens periodically to ensure cross-disciplinary synthesis
                new_persona = self.rotate_persona()
                next_location = f"{raw_next_location} [{new_persona}]"
                concept_blended = True
                print(f"🧬 Rotating to Persona Lens: '{new_persona}'", flush=True)
            else:
                next_location = raw_next_location

            step_data = {
                "step": step,
                "location": self.current_location,
                "persona": self.get_current_persona(),
                "temperature": float(self.current_temperature),
                "novelty_score": float(novelty_score),
                "coherence_score": float(coherence_score),
                "is_gold": is_gold,
                "leap_triggered": leap_triggered,
                "concept_blended": concept_blended,
                "response": response,
                "next_location": next_location,
                "energy_after_step": max(0, self.energy_level - 34),
                "pca_coord": latest_pca,
                "cohomological_dim": cohom_data["cohomological_dim"],
                "b0": cohom_data["b0"],
                "b1": cohom_data["b1"],
                "b2": cohom_data["b2"],
                "sheaf_obstruction": cohom_data["sheaf_obstruction"],
                "persistence_entropy": cohom_data["persistence_entropy"]
            }
            self.roadtrip_log.append(step_data)

            # Update live web dashboard with fully aligned telemetry
            if DASHBOARD_AVAILABLE and self.config.enable_dashboard:
                GLOBAL_STATE.update_step(step_data)
                GLOBAL_STATE.set_graph_data(
                    self.graph_engine.nodes,
                    self.graph_engine.edges,
                    self.graph_engine.get_pareto_frontier()
                )

            self.current_location = next_location
            self.energy_level -= 34
            step += 1

        print("\n==================================")
        print("🏆 EXPEDITION COMPLETE 🏆")
        print(f"Total Gold Veins Found: {len(self.gold_vault)}")
        print(f"Pareto Frontier Nodes: {len(self.graph_engine.get_pareto_frontier())}")
        print("==================================")
        
        if DASHBOARD_AVAILABLE and self.config.enable_dashboard:
            with GLOBAL_STATE.lock:
                GLOBAL_STATE.status = "COMPLETED"

        self.save_journal_to_json()

        if DASHBOARD_AVAILABLE and self.config.enable_dashboard:
            print(f"\n🖥️  Expedition complete. Web Dashboard remaining live at http://localhost:{self.config.port}")
            print("Press Ctrl+C to exit dashboard server.")
            try:
                while True:
                    time.sleep(1)
                    cmds = GLOBAL_STATE.pop_commands()
                    for cmd in cmds:
                        if cmd.get("action") == "RESTART":
                            self.reset_expedition_state()
                            self.start_expedition(steps, self.gold_threshold)
                            return
            except KeyboardInterrupt:
                print("\n👋 Server stopped.")
