import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Global shared state between Indiana Jones explorer and web dashboard
class ExpeditionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "IDLE" # IDLE, EXPEDITION_RUNNING, PAUSED, COMPLETED
        self.current_step = 0
        self.total_steps = 100
        self.current_location = "Uncharted"
        self.starting_concept = "Uncharted"
        self.temperature = 0.85
        self.energy = 100
        self.gold_threshold = 0.70
        self.momentum_weight = 0.5
        self.repulsion_strength = 5.0
        self.orthogonal_push_weight = 0.70
        self.current_persona = "Lateral Thinker"
        self.current_persona_idx = 0
        self.personas = []
        
        # Cohomological Dimensions & Persistent Cohomology State
        self.cohomological_dim = 1.0
        self.betti_0 = 1
        self.betti_1 = 0
        self.betti_2 = 0
        self.sheaf_obstruction = 0.0
        
        self.journal_history = []  # List of step dictionaries
        self.telemetry_history = []# Lightweight telemetry list without text
        self.gold_vault = []       # List of gold items
        self.frontier_nodes = []   # List of Pareto frontier node summaries
        self.graph_nodes = []      # List of node objects for visual graph
        self.graph_edges = []      # List of edge objects
        self.pca_coords = []       # List of {"x": float, "y": float, "label": str, "is_gold": bool, "step": int}
        self.control_commands = [] # List of queued command dicts (e.g. {"action": "WARP"})

    def set_personas(self, personas_list):
        with self.lock:
            self.personas = list(personas_list)

    def update_step(self, step_data):
        with self.lock:
            self.current_step = step_data.get("step", self.current_step)
            self.current_location = step_data.get("location", self.current_location)
            self.temperature = step_data.get("temperature", self.temperature)
            self.energy = step_data.get("energy_after_step", self.energy)
            self.current_persona = step_data.get("persona", self.current_persona)
            
            self.cohomological_dim = step_data.get("cohomological_dim", self.cohomological_dim)
            self.betti_0 = step_data.get("b0", self.betti_0)
            self.betti_1 = step_data.get("b1", self.betti_1)
            self.betti_2 = step_data.get("b2", self.betti_2)
            self.sheaf_obstruction = step_data.get("sheaf_obstruction", self.sheaf_obstruction)
            
            self.journal_history.append(step_data)
            
            # Append lightweight telemetry entry for chart rendering without text bloat
            self.telemetry_history.append({
                "step": step_data.get("step"),
                "novelty_score": step_data.get("novelty_score"),
                "coherence_score": step_data.get("coherence_score"),
                "cohomological_dim": step_data.get("cohomological_dim")
            })

            if step_data.get("is_gold"):
                self.gold_vault.append(step_data)
                
            if "pca_coord" in step_data:
                self.pca_coords.append(step_data["pca_coord"])

    def set_graph_data(self, nodes, edges, frontier):
        with self.lock:
            self.graph_nodes = nodes
            self.graph_edges = edges
            self.frontier_nodes = frontier

    def reset_expedition(self):
        self.current_step = 0
        self.current_location = self.starting_concept
        self.energy = 100
        self.cohomological_dim = 1.0
        self.betti_0 = 1
        self.betti_1 = 0
        self.betti_2 = 0
        self.sheaf_obstruction = 0.0
        self.journal_history.clear()
        self.telemetry_history.clear()
        self.gold_vault.clear()
        self.frontier_nodes.clear()
        self.graph_nodes.clear()
        self.graph_edges.clear()
        self.pca_coords.clear()
        self.status = "EXPEDITION_RUNNING"

    def pop_commands(self):
        with self.lock:
            cmds = list(self.control_commands)
            self.control_commands.clear()
            return cmds

    def add_command(self, cmd):
        with self.lock:
            action = cmd.get("action")
            if action == "UPDATE_PARAMS":
                if "gold_threshold" in cmd:
                    self.gold_threshold = float(cmd["gold_threshold"])
                if "orthogonal_push_weight" in cmd:
                    self.orthogonal_push_weight = float(cmd["orthogonal_push_weight"])
                if "repulsion_strength" in cmd:
                    self.repulsion_strength = float(cmd["repulsion_strength"])
                if "temperature" in cmd:
                    self.temperature = float(cmd["temperature"])
            elif action == "TELEPORT_CONCEPT":
                if "target_concept" in cmd and cmd["target_concept"]:
                    self.current_location = cmd["target_concept"].strip()
            elif action == "SET_PERSONA":
                if "persona_idx" in cmd:
                    self.current_persona_idx = int(cmd["persona_idx"])
                    if self.personas and 0 <= self.current_persona_idx < len(self.personas):
                        self.current_persona = self.personas[self.current_persona_idx]
            elif action == "PAUSE":
                self.status = "PAUSED"
            elif action == "RESUME":
                self.status = "EXPEDITION_RUNNING"
            elif action == "TOGGLE_PAUSE":
                if self.status == "PAUSED":
                    self.status = "EXPEDITION_RUNNING"
                else:
                    self.status = "PAUSED"
            elif action == "RESTART":
                self.reset_expedition()
            self.control_commands.append(cmd)

    def to_dict(self):
        with self.lock:
            return {
                "status": self.status,
                "is_paused": (self.status == "PAUSED"),
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "current_location": self.current_location,
                "starting_concept": self.starting_concept,
                "temperature": self.temperature,
                "energy": self.energy,
                "gold_threshold": self.gold_threshold,
                "momentum_weight": self.momentum_weight,
                "repulsion_strength": self.repulsion_strength,
                "orthogonal_push_weight": self.orthogonal_push_weight,
                "current_persona": self.current_persona,
                "current_persona_idx": self.current_persona_idx,
                "personas": self.personas,
                "cohomological_dim": self.cohomological_dim,
                "betti_0": self.betti_0,
                "betti_1": self.betti_1,
                "betti_2": self.betti_2,
                "sheaf_obstruction": self.sheaf_obstruction,
                "gold_count": len(self.gold_vault),
                "journal_count": len(self.journal_history),
                "journal_history": self.journal_history, # Full journal history for research inspection
                "telemetry_history": self.telemetry_history[-300:], # Cap lightweight telemetry
                "gold_vault": self.gold_vault, # Full gold vault items
                "frontier_nodes": self.frontier_nodes,
                "graph_nodes": self.graph_nodes,
                "graph_edges": self.graph_edges,
                "pca_coords": self.pca_coords # Full PCA coordinates
            }

    def load_journal_data(self, data, filename="", mode="replace"):
        with self.lock:
            starting_concept = data.get("starting_concept", "Historical Expedition")
            raw_entries = data.get("journal_history") or data.get("journal_entries") or []
            raw_gold = data.get("gold_vault") or []
            session_label = data.get("starting_concept") or (os.path.basename(filename) if filename else "Imported Expedition")

            # Determine starting step index if merging
            start_step_offset = len(self.journal_history) if mode == "merge" else 0

            new_history = []
            new_gold = []
            
            for idx, item in enumerate(raw_entries, 1):
                step_idx = start_step_offset + idx
                if isinstance(item, str):
                    entry = {
                        "step": step_idx,
                        "location": f"Step {idx}",
                        "response": item,
                        "persona": "Historical Exploration",
                        "novelty_score": 0.75,
                        "coherence_score": 8,
                        "cohomological_dim": 1.0,
                        "betti_0": 1,
                        "betti_1": 0,
                        "betti_2": 0,
                        "is_gold": (item in raw_gold) or (idx in [1, 5, 10]),
                        "next_location": "Latent Space",
                        "session_source": session_label
                    }
                elif isinstance(item, dict):
                    entry = dict(item)
                    if mode == "merge":
                        entry["step"] = step_idx
                    entry["session_source"] = entry.get("session_source") or session_label
                else:
                    continue

                new_history.append(entry)
                if entry.get("is_gold"):
                    new_gold.append(entry)

            if mode == "replace":
                self.journal_history = new_history
                self.gold_vault = new_gold
                self.starting_concept = starting_concept
                self.current_location = data.get("final_location") or starting_concept
                self.total_steps = len(new_history)
                self.current_step = len(new_history)
                self.status = "HISTORICAL_ARCHIVE"
                # Rebuild telemetry history from loaded entries
                self.telemetry_history = [{
                    "step": e.get("step"),
                    "novelty_score": e.get("novelty_score", 0.75),
                    "coherence_score": e.get("coherence_score", 8),
                    "cohomological_dim": e.get("cohomological_dim", 1.0)
                } for e in new_history]
            else: # merge
                self.journal_history.extend(new_history)
                self.gold_vault.extend(new_gold)
                self.total_steps = len(self.journal_history)
                self.current_step = len(self.journal_history)
                for e in new_history:
                    self.telemetry_history.append({
                        "step": e.get("step"),
                        "novelty_score": e.get("novelty_score", 0.75),
                        "coherence_score": e.get("coherence_score", 8),
                        "cohomological_dim": e.get("cohomological_dim", 1.0)
                    })

            return len(new_history)

GLOBAL_STATE = ExpeditionState()

def list_available_journals():
    cwd = os.path.dirname(__file__) or "."
    journal_files = []
    for fname in os.listdir(cwd):
        if fname.startswith("journal") and fname.endswith(".json"):
            filepath = os.path.join(cwd, fname)
            try:
                stat = os.stat(filepath)
                size_kb = round(stat.st_size / 1024, 1)
                mtime = stat.st_mtime
                import datetime
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                
                # Peek inside file for quick metadata
                starting_concept = "Historical Expedition"
                step_count = 0
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    starting_concept = content.get("starting_concept") or "Historical Run"
                    entries = content.get("journal_history") or content.get("journal_entries") or []
                    step_count = len(entries)
                    
                journal_files.append({
                    "filename": fname,
                    "starting_concept": starting_concept,
                    "step_count": step_count,
                    "size_kb": size_kb,
                    "modified_time": mtime_str,
                    "mtime_ts": mtime
                })
            except Exception as e:
                pass
    journal_files.sort(key=lambda x: x["mtime_ts"], reverse=True)
    return journal_files

class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress standard HTTP access logging to keep terminal clean
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = GLOBAL_STATE.to_dict()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        elif parsed.path == "/api/journals":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            journals = list_available_journals()
            self.wfile.write(json.dumps({"journals": journals}, ensure_ascii=False).encode('utf-8'))
        elif parsed.path == "/api/step":
            query = parse_qs(parsed.query)
            step_num = int(query.get("step", [0])[0])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            step_data = None
            with GLOBAL_STATE.lock:
                for entry in GLOBAL_STATE.journal_history:
                    if entry.get("step") == step_num:
                        step_data = entry
                        break
            self.wfile.write(json.dumps(step_data or {"error": "Step not found"}, ensure_ascii=False).encode('utf-8'))
        elif parsed.path in ["/", "/index.html", "/dashboard.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            dashboard_file = os.path.join(os.path.dirname(__file__), "dashboard.html")
            if os.path.exists(dashboard_file):
                with open(dashboard_file, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                self.wfile.write(b"<h1>Dashboard file not found</h1>")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/control":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode('utf-8'))
                action = payload.get("action")
                if action:
                    GLOBAL_STATE.add_command(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "received": action}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        elif parsed.path == "/api/load_journal":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode('utf-8'))
                filename = payload.get("filename", "")
                mode = payload.get("mode", "replace") # "replace" or "merge"
                custom_json = payload.get("custom_json")
                
                cwd = os.path.dirname(__file__) or "."
                loaded_count = 0
                
                if custom_json and isinstance(custom_json, dict):
                    loaded_count = GLOBAL_STATE.load_journal_data(custom_json, filename=filename or "Uploaded File", mode=mode)
                elif filename:
                    filepath = os.path.join(cwd, os.path.basename(filename))
                    if os.path.exists(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            loaded_count = GLOBAL_STATE.load_journal_data(data, filename=filename, mode=mode)
                    else:
                        raise FileNotFoundError(f"File {filename} not found")
                else:
                    raise ValueError("Neither filename nor custom_json provided")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "loaded_count": loaded_count,
                    "mode": mode,
                    "total_journal_count": len(GLOBAL_STATE.journal_history)
                }, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_dashboard_server(port=8000):
    server = HTTPServer(("0.0.0.0", port), DashboardHTTPRequestHandler)
    print(f"🖥️  Expedition Web Dashboard live at: http://localhost:{port}")
    server.serve_forever()

def start_server_in_thread(port=8000):
    t = threading.Thread(target=run_dashboard_server, args=(port,), daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    run_dashboard_server(8000)
