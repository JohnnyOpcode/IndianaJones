import argparse
import os
from dataclasses import dataclass


@dataclass
class ExplorerConfig:
    model_path: str = "./LLM/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    starting_concept: str = "pancreatic cancer comorbidities"
    steps: int = 100
    gold_threshold: float = 0.90
    port: int = 8000
    enable_dashboard: bool = True
    output_dir: str = "."
    temperature: float = 0.90


def parse_args() -> ExplorerConfig:
    defaults = ExplorerConfig()
    parser = argparse.ArgumentParser(
        description="Indiana Jones: Latent Space Explorer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default=os.environ.get(
            "MODEL_PATH", defaults.model_path
        ),
        help="Path to the GGUF model file.",
    )
    parser.add_argument(
        "--concept",
        "-c",
        type=str,
        default=defaults.starting_concept,
        help="Starting concept phrase to drop into latent space.",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=defaults.steps,
        help="Number of expedition steps to execute.",
    )
    parser.add_argument(
        "--gold-threshold",
        "-g",
        type=float,
        default=defaults.gold_threshold,
        help="Novelty threshold for gold vein appraisal.",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=defaults.port,
        help="Port for the live web dashboard HTTP server.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable running the live web dashboard server.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=defaults.output_dir,
        help="Directory to save expedition journal JSON files.",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=defaults.temperature,
        help="Initial LLM sampling temperature.",
    )

    args = parser.parse_args()

    return ExplorerConfig(
        model_path=args.model_path,
        starting_concept=args.concept,
        steps=args.steps,
        gold_threshold=args.gold_threshold,
        port=args.port,
        enable_dashboard=not args.no_dashboard if args.no_dashboard else defaults.enable_dashboard,
        output_dir=args.output_dir,
        temperature=args.temperature,
    )
