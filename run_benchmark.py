from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(dataset: str = "data/hotpot_100.json", out_dir: str = "outputs/sample_run", reflexion_attempts: int = 3) -> None:
    examples = load_dataset(dataset)
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    react_runs_file = out_path / "react_runs.jsonl"
    reflexion_runs_file = out_path / "reflexion_runs.jsonl"
    
    from src.reflexion_lab.schemas import RunRecord
    
    # --- Chạy ReAct Agent có resume ---
    react_records = []
    if react_runs_file.exists():
        with react_runs_file.open("r", encoding="utf-8") as f:
            for line in f:
                react_records.append(RunRecord.model_validate_json(line))
                
    react_done_ids = {r.qid for r in react_records}
    for example in examples:
        if example.qid not in react_done_ids:
            record = react.run(example)
            react_records.append(record)
            with react_runs_file.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
                
    # --- Chạy Reflexion Agent có resume ---
    reflexion_records = []
    if reflexion_runs_file.exists():
        with reflexion_runs_file.open("r", encoding="utf-8") as f:
            for line in f:
                reflexion_records.append(RunRecord.model_validate_json(line))
                
    reflexion_done_ids = {r.qid for r in reflexion_records}
    for example in examples:
        if example.qid not in reflexion_done_ids:
            record = reflexion.run(example)
            reflexion_records.append(record)
            with reflexion_runs_file.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")

    all_records = react_records + reflexion_records
    report = build_report(all_records, dataset_name=Path(dataset).name, mode="live")
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
