from __future__ import annotations
import argparse
import runpy
import sys
from pathlib import Path
import os

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def run_script(script_rel_path: str) -> None:
    project_root = get_project_root()
    script_path = project_root / "src" / "visualization" / script_rel_path
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    os.chdir(project_root)

    original_argv = sys.argv
    try:
        sys.argv = [script_path.name]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = original_argv

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizations orchestrator (scatter, histogram, pair-plot)."
    )
    parser.add_argument(
        "which",
        choices=["scatter", "hist", "pair", "all"],
        help="Which visualization to run."
    )
    args = parser.parse_args()

    mapping = {
        "scatter": "scatter_plot/scatter_plot.py",
        "hist": "histogram/histogram.py",
        "pair": "pair_plot/pair_plot.py",
    }

    if args.which == "all":
        for key in ("hist", "scatter", "pair"):
            print(f"\n=== Running: {key} ===")
            try:
                run_script(mapping[key])
            except SystemExit as e:
                if e.code != 0:
                    print(f"[ERROR] {key} failed with exit code {e.code}")
                    sys.exit(e.code)
            except Exception as e:
                print(f"[ERROR] {key} failed: {e}")
                sys.exit(1)
        return

    run_script(mapping[args.which])

if __name__ == "__main__":
    main()