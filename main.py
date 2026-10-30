# main.py
import json
import sys
import time
from pathlib import Path

from runtime.message_bus import MessageBus
from agents.base import LLMClient, OllamaConfig
from agents.data_ingestion import DataIngestionAgent
from agents.planner import PlannerAgent
from agents.executor import AnalysisExecutorAgent
from agents.critic import CritiqueAgent
from agents.reporter import ReportingAgent


def main(csv_path: str):
    base_dir = Path(__file__).resolve().parent
    run_dir = base_dir / "reports" / time.strftime("run_%Y%m%d_%H%M%S")
    figs_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    bus = MessageBus(log_dir=run_dir)
    llm_client = LLMClient(OllamaConfig(model="llama3.1:8b"))

    timings = {}

    print("[MAIN] Starting Data Ingestion...")
    t0 = time.time()
    DataIngestionAgent("DataIngestion", llm_client, bus).run(csv_path)
    timings["data_ingestion"] = time.time() - t0
    print(f"[MAIN] Data Ingestion done in {timings['data_ingestion']:.1f}s")

    print("[MAIN] Starting Planner...")
    t0 = time.time()
    PlannerAgent("Planner", llm_client, bus).run()
    timings["planner"] = time.time() - t0
    print(f"[MAIN] Planner done in {timings['planner']:.1f}s")

    print("[MAIN] Running Executor (deterministic EDA+ML)...")
    t0 = time.time()
    AnalysisExecutorAgent("Executor", llm_client, bus, figures_dir=figs_dir).run()
    timings["executor"] = time.time() - t0
    print(f"[MAIN] Executor done in {timings['executor']:.1f}s")

    # Critic is optional; keep it but it cannot crash the run now
    print("[MAIN] Critic...")
    t0 = time.time()
    CritiqueAgent("Critic", llm_client, bus).run()
    timings["critic"] = time.time() - t0
    print(f"[MAIN] Critic done in {timings['critic']:.1f}s")

    print("[MAIN] Reporting...")
    t0 = time.time()
    report_path = ReportingAgent(
        "Reporter", llm_client, bus, templates_dir=base_dir / "templates"
    ).run(report_dir=run_dir)
    timings["reporter"] = time.time() - t0

    (run_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"[MAIN] Done. Report: {report_path}")
    print(f"[MAIN] Figures: {figs_dir}")
    print(f"[MAIN] Events: {run_dir / 'events.jsonl'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/data.csv")
        raise SystemExit(1)
    main(sys.argv[1])
