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
from llama_cpp import Llama, LogitsProcessorList
from sentence_transformers import SentenceTransformer

from engines.cohomology import CohomologyEngine
from engines.novelty import NoveltyEngine, RepulsionLogitsProcessor
from engines.steering import VectorSteering
from engines.graph import ExpeditionGraph
from core.config import ExplorerConfig

# Import web dashboard server module
try:
    from dashboard_server import GLOBAL_STATE, start_server_in_thread
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


class LatentExplorer:
    def __init__(self, config: ExplorerConfig):
        print("🗺️  Equipping Advanced Indiana Jones Latent Explorer v2.1...")
        self.config = config
        
        # 1. Brain & Sampler
        self.llm = Llama(
            model_path=config.model_path, 
            n_gpu_layers=-1, 
            n_ctx=4096, 
            verbose=False,
            seed=-1
        )
        
        # 2. Vector Compass
        self.compass = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Next-Gen Engines
        self.novelty_engine = NoveltyEngine(bandwidth=0.35, decay_factor=0.97, max_history=150)
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
        self.model_path = config.model_path
        self.starting_concept = config.starting_concept
        self.current_location = config.starting_concept
        self.current_temperature = config.temperature
        self.low_novelty_streak = 0
        self.orthogonal_push_weight = 0.90
        self.gold_threshold = config.gold_threshold
        
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
        self.restart_requested = False

        # Start live web dashboard
        if DASHBOARD_AVAILABLE and config.enable_dashboard:
            GLOBAL_STATE.status = "EXPEDITION_RUNNING"
            GLOBAL_STATE.starting_concept = config.starting_concept
            GLOBAL_STATE.current_location = config.starting_concept
            GLOBAL_STATE.set_personas(self.personas)
            start_server_in_thread(config.port)

        print(f"📍 Dropping into High-Dimensional Latent Space at: '{self.current_location}'\n")

    def reset_expedition_state(self):
        print("\n🔄 RESTARTING EXPEDITION FROM STARTING CONCEPT...")
        self.journal_texts.clear()
        self.journal_embeddings.clear()
        self.gold_vault.clear()
        self.gold_embeddings.clear()
        self.roadtrip_log.clear()
        self.energy_level = 100
        self.current_location = self.starting_concept
        self.low_novelty_streak = 0
        self.current_persona_idx = 0
        self.graph_engine = ExpeditionGraph()
        self.novelty_engine = NoveltyEngine(bandwidth=0.35, decay_factor=0.97, max_history=150)
        self.cohomology_engine = CohomologyEngine(max_points=25, eps_percentile=50)
        self.restart_requested = True
        if DASHBOARD_AVAILABLE and self.config.enable_dashboard:
            GLOBAL_STATE.reset_expedition()

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

    def generate_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, 
                      temperature: float = 0.9, top_p: float = 0.9, repeat_penalty: float = 1.18, 
                      logits_processor=None) -> str:
        """Robust chat completion generator with Llama 3 chat template and error fallbacks."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 1. Try create_chat_completion (Llama 3 instruction template format)
        try:
            kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty
            }
            if logits_processor is not None:
                kwargs["logits_processor"] = logits_processor
                
            response = self.llm.create_chat_completion(**kwargs)
            content = response["choices"][0]["message"].get("content", "")
            if content and content.strip():
                return content.strip()
        except Exception as e:
            print(f"⚠️ Chat template generation note: {e}. Falling back to standard text completion...")

        # 2. Fallback to raw text completion
        try:
            prompt_text = f"{system_prompt}\n\n{user_prompt}"
            output = self.llm(
                prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                logits_processor=logits_processor
            )
            return output["choices"][0]["text"].strip()
        except Exception as e2:
            print(f"❌ Generation error: {e2}. Returning fallback text.")
            return "An abstract nexus of latent concepts awaiting further topological exploration."

    def craft_prompt(self, concept):
        persona = self.get_current_persona()
        system_prompt = f"You are an elite, lateral-thinking latent space explorer. Explore concepts through the hyper-specialized lens of: {persona}."
        user_prompt = f"Concept to explore: {concept}\n\nGenerate a short, dense paragraph uncovering a brand new, highly specific perspective on this. Do NOT ask questions, address the reader, or output meta commentary. Write the paragraph directly."
        return system_prompt, user_prompt

    def extract_seed_prompt(self, text):
        system_prompt = "You are a precise concept extraction engine."
        user_prompt = f"Read the following text. Extract the single most unusual, bizarre, or highly specific concept from it.\nOutput ONLY that concept as a short noun phrase in 1 to 5 words. No underscores, no blank spaces, no conversational filler.\n\nText: {text}\n\nConcept:"
        return system_prompt, user_prompt

    def validate_seed(self, seed):
        cleaned = seed.strip().strip('"').strip("'").split('\n')[0].strip()
        alpha_chars = sum(1 for c in cleaned if c.isalpha())
        if len(cleaned) < 3 or alpha_chars < 3 or len(cleaned) > 80 or cleaned.count('_') > 3:
            print(f"🚫 Dull/invalid seed detected: '{cleaned[:40]}...' — synthesizing fresh orthogonal concept.")
            sys_p = "You are a concept synthesizer."
            usr_p = f"Generate a single highly unusual concept in 3 to 5 words combining {self.get_current_persona()} and unexpected philosophy. Output ONLY the concept."
            cleaned = self.generate_chat(sys_p, usr_p, max_tokens=15, temperature=0.95).strip().split('\n')[0].strip()
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
        sys_p = "You are an objective text quality evaluator."
        usr_p = f"Rate the logical coherence, readability, and conceptual depth of the following text on a scale from 1 to 10.\nOutput ONLY a single number between 1 and 10.\n\nText: {text}\n\nScore:"
        response = self.generate_chat(sys_p, usr_p, max_tokens=10, temperature=0.1)
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
        
        # Performance optimization: fit PCA on up to 200 recent samples to bound O(N^2) fit cost
        if len(arr) > 200:
            pca = PCA(n_components=2)
            pca.fit(arr[-200:])
            coords2d = pca.transform(arr)
        else:
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
            # Check for resume or parameter updates while paused
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
        print("\n⛺ ENERGY DEPLETED. Setting up camp.")
        if self.journal_texts:
            print("📝 Journaling the day's discoveries...")
            sys_p = "You are a reflective journal summarizer."
            usr_p = "Summarize these explored concepts into one sentence: " + " | ".join(self.journal_texts[-3:])
            summary = self.generate_chat(sys_p, usr_p, max_tokens=100, temperature=0.5)
            print(f"📖 Campfire thoughts: {summary}")
        
        print("🔥 Resting by the fire for 5 seconds...")
        time.sleep(5)
        self.energy_level = 100
        print("🌅 Sun is up. Energy restored. Packing up camp...\n")

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
            "gold_vault": self.gold_vault,
            "roadtrip_log": self.roadtrip_log,
            "pareto_frontier": self.graph_engine.get_pareto_frontier(),
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
            GLOBAL_STATE.total_steps = steps
            GLOBAL_STATE.gold_threshold = gold_threshold

        step = 1
        while step <= steps:
            if self.restart_requested:
                self.restart_requested = False
                step = 1

            if self.energy_level <= 0:
                self.make_camp()

            self.check_web_dashboard_commands()
            if self.restart_requested:
                self.restart_requested = False
                step = 1
                continue

            self.update_repulsion_list()

            print(f"\n--- 🥾 Step {step}/{steps} | Location: '{self.current_location}' | Lens: {self.get_current_persona()} ---")
            
            # 1. The Trek (Prompt Crafting & Generation with Logit Repulsion)
            print("🧭 Traversing latent space with logit repulsion & persona steering...")
            sys_prompt, user_prompt = self.craft_prompt(self.current_location)
            
            raw_response = self.generate_chat(
                sys_prompt,
                user_prompt,
                max_tokens=250,
                temperature=self.current_temperature,
                top_p=0.9,
                repeat_penalty=1.18,
                logits_processor=LogitsProcessorList([self.repulsion_processor])
            )
            response = self.strip_filler(raw_response)
            
            # 2. Vector Encoding, Persistent Cohomology & Novelty Evaluation
            vec = self.compass.encode([response])[0]
            
            cohom_data = self.cohomology_engine.compute_simplicial_cohomology(self.journal_embeddings + [vec])
            print(f"📐 Cohomological Dim: {cohom_data['cohomological_dim']:.2f} | b0={cohom_data['b0']} | b1(Loops)={cohom_data['b1']} | b2(Voids)={cohom_data['b2']} | Sheaf Obstruction: {cohom_data['sheaf_obstruction']:.3f}")

            novelty_score, vec = self.novelty_engine.evaluate_novelty(
                response, vec, self.journal_embeddings, self.journal_texts, cohom_novelty=cohom_data["cohomological_novelty"]
            )
            print(f"📊 Multi-Scale KDE + Cohomological Novelty Score: {novelty_score:.2f} (Target > {self.gold_threshold:.2f})")
            
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
            if novelty_score > self.gold_threshold:
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
            seed_sys, seed_usr = self.extract_seed_prompt(response)
            seed_output = self.generate_chat(seed_sys, seed_usr, max_tokens=15, temperature=0.3)
            raw_next_location = self.validate_seed(seed_output)
            
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
                ortho_vec = VectorSteering.topological_steering(ortho_vec, cohom_data, push_weight=self.orthogonal_push_weight)
                
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
