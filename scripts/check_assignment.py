from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "LC-01": "agents/llm_client.py",
    "LC-02": "tools/langchain_tools.py",
    "LC-03": "agents/memory.py",
    "LC-04": "tools/rag.py",
    "LG-01": "agents/state.py",
    "LG-02": "agents/graph.py",
    "LG-03": "agents/graph.py",
    "LG-04": "agents/graph.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Show implementation status for the course TODOs.")
    parser.add_argument("--strict", action="store_true", help="exit with code 1 while required TODO markers remain")
    args = parser.parse_args()

    incomplete: list[str] = []
    for task_id, relative_path in TASKS.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        marker_present = f"TODO {task_id}" in text
        status = "TODO" if marker_present else "IMPLEMENTED"
        print(f"{task_id:5} {status:11} {relative_path}")
        if marker_present:
            incomplete.append(task_id)

    print()
    if incomplete:
        print("Remaining tasks: " + ", ".join(incomplete))
        print("Run: python -B -m unittest discover -s tests -v")
        return 1 if args.strict else 0
    print("No assignment markers remain. Run the full test suite and a browser smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
