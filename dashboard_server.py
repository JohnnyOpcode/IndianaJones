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
        self.gold_threshold = 0.55
        self.momentum_weight = 0.5
        self.repulsion_strength = 5.0
        self.current_persona = "Lateral Thinker"
        
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

    def pop_commands(self):
        with self.lock:
            cmds = list(self.control_commands)
            self.control_commands.clear()
            return cmds

    def add_command(self, cmd):
        with self.lock:
            self.control_commands.append(cmd)

    def to_dict(self):
        with self.lock:
            return {
                "status": self.status,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "current_location": self.current_location,
                "starting_concept": self.starting_concept,
                "temperature": self.temperature,
                "energy": self.energy,
                "gold_threshold": self.gold_threshold,
                "momentum_weight": self.momentum_weight,
                "repulsion_strength": self.repulsion_strength,
                "current_persona": self.current_persona,
                "cohomological_dim": self.cohomological_dim,
                "betti_0": self.betti_0,
                "betti_1": self.betti_1,
                "betti_2": self.betti_2,
                "sheaf_obstruction": self.sheaf_obstruction,
                "gold_count": len(self.gold_vault),
                "journal_count": len(self.journal_history),
                "journal_history": self.journal_history[-15:], # Last 15 for feed
                "telemetry_history": self.telemetry_history[-150:], # Cap lightweight telemetry to last 150
                "gold_vault": self.gold_vault[-25:], # Last 25 gold items
                "frontier_nodes": self.frontier_nodes,
                "graph_nodes": self.graph_nodes[-30:],
                "graph_edges": self.graph_edges[-40:],
                "pca_coords": self.pca_coords[-150:] # Cap PCA payload to last 150 points for low bandwidth
            }

GLOBAL_STATE = ExpeditionState()

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
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
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
