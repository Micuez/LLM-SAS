#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize MILP decision trace JSONL files.")
    parser.add_argument(
        "--trace-dir",
        default="results/traces",
        help="Directory containing decision trace .jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated CSV summaries. Defaults to the trace directory.",
    )
    return parser.parse_args()


def load_trace_records(trace_dir):
    trace_dir = Path(trace_dir)
    records = []
    for trace_file in sorted(trace_dir.glob("*.jsonl")):
        with open(trace_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                payload["_trace_file"] = trace_file.name
                payload["_line_no"] = line_no
                records.append(payload)
    return records


def flatten_round_record(record):
    llm_decision = record.get("llm_decision", {}) or {}
    bandit_decision = record.get("bandit_decision", {}) or {}
    checked_decision = record.get("checked_decision", {}) or {}
    structural_signals = record.get("structural_signals", {}) or {}
    global_features = record.get("global_features", {}) or {}

    return {
        "trace_file": record.get("_trace_file"),
        "line_no": record.get("_line_no"),
        "problem_code": record.get("problem_code"),
        "instance_idx": record.get("instance_idx"),
        "round_idx": record.get("round_idx"),
        "selected_operator": record.get("selected_operator"),
        "llm_operator": llm_decision.get("operator"),
        "bandit_operator": bandit_decision.get("operator"),
        "checked_operator": checked_decision.get("operator"),
        "bandit_override": bandit_decision.get("bandit_override"),
        "checker_reason": checked_decision.get("checker_reason"),
        "focus": checked_decision.get("focus", llm_decision.get("focus")),
        "exploration_level": checked_decision.get("exploration_level", llm_decision.get("exploration_level")),
        "free_ratio": checked_decision.get("free_ratio", bandit_decision.get("free_ratio", llm_decision.get("free_ratio"))),
        "time_budget": checked_decision.get("time_budget", bandit_decision.get("time_budget", llm_decision.get("time_budget"))),
        "released_count": record.get("released_count"),
        "released_ratio": record.get("released_ratio"),
        "solver_runtime": record.get("solver_runtime"),
        "solver_status": record.get("solver_status"),
        "objective_before": record.get("objective_before"),
        "objective_after": record.get("objective_after"),
        "improvement": record.get("improvement"),
        "elapsed_time": record.get("elapsed_time"),
        "timeout_count": record.get("timeout_count"),
        "lp_gap": global_features.get("lp_gap"),
        "incumbent_obj": global_features.get("incumbent_obj"),
        "no_improve_rounds": global_features.get("no_improve_rounds"),
        "fractionality_concentration": structural_signals.get("fractionality_concentration"),
        "tight_constraint_ratio": structural_signals.get("tight_constraint_ratio"),
        "high_objective_variable_concentration": structural_signals.get("high_objective_variable_concentration"),
        "graph_block_modularity": structural_signals.get("graph_block_modularity"),
        "recent_explored_variable_ratio": structural_signals.get("recent_explored_variable_ratio"),
    }


def aggregate_operator_stats(records):
    grouped = defaultdict(lambda: {
        "problem_codes": set(),
        "instances": set(),
        "rounds": 0,
        "success_rounds": 0,
        "timeout_rounds": 0,
        "bandit_override_rounds": 0,
        "total_improvement": 0.0,
        "total_runtime": 0.0,
        "total_released_ratio": 0.0,
    })

    for record in records:
        operator = record.get("selected_operator") or "UNKNOWN"
        entry = grouped[operator]
        entry["problem_codes"].add(record.get("problem_code"))
        entry["instances"].add(record.get("instance_idx"))
        entry["rounds"] += 1
        if record.get("solver_status") == "success":
            entry["success_rounds"] += 1
        if record.get("solver_status") == "timeout_or_failure":
            entry["timeout_rounds"] += 1
        if record.get("bandit_override"):
            entry["bandit_override_rounds"] += 1
        entry["total_improvement"] += float(record.get("improvement") or 0.0)
        entry["total_runtime"] += float(record.get("solver_runtime") or 0.0)
        entry["total_released_ratio"] += float(record.get("released_ratio") or 0.0)

    summary_rows = []
    for operator, entry in sorted(grouped.items()):
        rounds = max(entry["rounds"], 1)
        avg_runtime = entry["total_runtime"] / rounds
        summary_rows.append({
            "selected_operator": operator,
            "problem_codes": ",".join(sorted(str(x) for x in entry["problem_codes"] if x is not None)),
            "instance_count": len(entry["instances"]),
            "rounds": entry["rounds"],
            "success_rate": entry["success_rounds"] / rounds,
            "timeout_rate": entry["timeout_rounds"] / rounds,
            "bandit_override_rate": entry["bandit_override_rounds"] / rounds,
            "avg_improvement": entry["total_improvement"] / rounds,
            "avg_runtime": avg_runtime,
            "avg_released_ratio": entry["total_released_ratio"] / rounds,
            "avg_improvement_per_second": entry["total_improvement"] / max(entry["total_runtime"], 1e-9),
        })
    return summary_rows


def aggregate_checker_reasons(records):
    counter = defaultdict(int)
    for record in records:
        reason = record.get("checker_reason") or "none"
        counter[reason] += 1
    rows = []
    total = sum(counter.values())
    for reason, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append({
            "checker_reason": reason,
            "count": count,
            "ratio": count / max(total, 1),
        })
    return rows


def markdown_table(rows, columns):
    if not rows:
        return "_No data available._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def build_operator_report(round_rows, operator_rows, checker_rows, trace_dir):
    total_rounds = len(round_rows)
    success_rounds = sum(1 for row in round_rows if row.get("solver_status") == "success")
    timeout_rounds = sum(1 for row in round_rows if row.get("solver_status") == "timeout_or_failure")
    bandit_override_rounds = sum(1 for row in round_rows if row.get("bandit_override"))
    avg_improvement = sum(float(row.get("improvement") or 0.0) for row in round_rows) / max(total_rounds, 1)
    avg_runtime = sum(float(row.get("solver_runtime") or 0.0) for row in round_rows) / max(total_rounds, 1)

    top_operators = sorted(
        operator_rows,
        key=lambda row: (-float(row.get("avg_improvement_per_second", 0.0)), -float(row.get("success_rate", 0.0))),
    )[:5]
    timeout_operators = sorted(
        operator_rows,
        key=lambda row: (-float(row.get("timeout_rate", 0.0)), -int(row.get("rounds", 0))),
    )[:5]

    lines = []
    lines.append("# Operator Report")
    lines.append("")
    lines.append(f"Trace directory: `{trace_dir}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Total rounds: `{total_rounds}`")
    lines.append(f"- Success rounds: `{success_rounds}`")
    lines.append(f"- Timeout/failure rounds: `{timeout_rounds}`")
    lines.append(f"- Bandit override rounds: `{bandit_override_rounds}`")
    lines.append(f"- Average improvement per round: `{avg_improvement:.4f}`")
    lines.append(f"- Average solver runtime per round: `{avg_runtime:.4f}`")
    lines.append("")
    lines.append("## Operator Summary")
    lines.append("")
    lines.append(markdown_table(
        operator_rows,
        [
            "selected_operator",
            "rounds",
            "success_rate",
            "timeout_rate",
            "bandit_override_rate",
            "avg_improvement",
            "avg_runtime",
            "avg_released_ratio",
            "avg_improvement_per_second",
        ],
    ))
    lines.append("")
    lines.append("## Best Operators by Improvement per Second")
    lines.append("")
    lines.append(markdown_table(
        top_operators,
        [
            "selected_operator",
            "rounds",
            "success_rate",
            "avg_improvement",
            "avg_runtime",
            "avg_improvement_per_second",
        ],
    ))
    lines.append("")
    lines.append("## Most Timeout-Prone Operators")
    lines.append("")
    lines.append(markdown_table(
        timeout_operators,
        [
            "selected_operator",
            "rounds",
            "timeout_rate",
            "avg_runtime",
            "avg_released_ratio",
        ],
    ))
    lines.append("")
    lines.append("## Checker Reasons")
    lines.append("")
    lines.append(markdown_table(
        checker_rows,
        ["checker_reason", "count", "ratio"],
    ))
    lines.append("")
    return "\n".join(lines)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    args = parse_args()
    trace_dir = Path(args.trace_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else trace_dir

    records = load_trace_records(trace_dir)
    round_rows = [flatten_round_record(record) for record in records]
    operator_rows = aggregate_operator_stats(round_rows)
    checker_rows = aggregate_checker_reasons(round_rows)
    report_content = build_operator_report(round_rows, operator_rows, checker_rows, trace_dir)

    write_csv(output_dir / "decision_trace_rounds.csv", round_rows)
    write_csv(output_dir / "decision_trace_operator_summary.csv", operator_rows)
    write_csv(output_dir / "decision_trace_checker_summary.csv", checker_rows)
    write_text(output_dir / "operator_report.md", report_content)

    print(f"Loaded {len(records)} trace records from {trace_dir}")
    print(f"Wrote round-level summary to {output_dir / 'decision_trace_rounds.csv'}")
    print(f"Wrote operator summary to {output_dir / 'decision_trace_operator_summary.csv'}")
    print(f"Wrote checker summary to {output_dir / 'decision_trace_checker_summary.csv'}")
    print(f"Wrote markdown report to {output_dir / 'operator_report.md'}")


if __name__ == "__main__":
    main()
