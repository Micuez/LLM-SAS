#!/usr/bin/env python3

import argparse
import importlib
import json
import os
from pathlib import Path


REQUIRED_MODULES = ["numpy", "requests", "joblib", "gurobipy"]
DATASET_RELATIVE_PATHS = {
    "MIKS": "MIKS_easy_instance/LP",
    "IS": "IS_easy_instance/LP",
    "SC": "SC_easy_instance/LP",
    "MVC": "MVC_easy_instance/LP",
}
ENV_KEYS = [
    "LLM_SAS_MILP_DATA_ROOT",
    "LLM_SAS_LLM_ENDPOINT",
    "LLM_SAS_LLM_API_KEY",
    "LLM_SAS_LLM_MODEL",
    "LLM_SAS_LLM_BACKEND",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke check for the LLM-SAS MILP pipeline.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        help="Path to the LLM-SAS project root.",
    )
    parser.add_argument(
        "--check-endpoint",
        action="store_true",
        help="Probe the configured OpenAI-compatible endpoint when requests and endpoint are available.",
    )
    return parser.parse_args()


def check_modules():
    rows = []
    requests_module = None
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            rows.append({"module": name, "ok": True, "detail": getattr(module, "__version__", "imported")})
            if name == "requests":
                requests_module = module
        except Exception as exc:
            rows.append({"module": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
    return rows, requests_module


def resolve_dataset_root(project_root: Path):
    env_root = os.environ.get("LLM_SAS_MILP_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve(), "env"
    return project_root.resolve(), "project"


def check_datasets(project_root: Path):
    dataset_root, source = resolve_dataset_root(project_root)
    rows = []
    for problem_code, rel_path in DATASET_RELATIVE_PATHS.items():
        dataset_path = (dataset_root / rel_path).resolve()
        lp_files = sorted(dataset_path.glob("*.lp")) if dataset_path.exists() else []
        rows.append(
            {
                "problem_code": problem_code,
                "path": str(dataset_path),
                "exists": dataset_path.exists(),
                "lp_count": len(lp_files),
                "source": source,
            }
        )
    return rows


def check_env():
    return [{"key": key, "value": os.environ.get(key)} for key in ENV_KEYS]


def check_endpoint(requests_module):
    endpoint = os.environ.get("LLM_SAS_LLM_ENDPOINT")
    model = os.environ.get("LLM_SAS_LLM_MODEL")
    api_key = os.environ.get("LLM_SAS_LLM_API_KEY", "EMPTY")

    if requests_module is None:
        return {"ok": False, "detail": "requests is not installed"}
    if not endpoint:
        return {"ok": False, "detail": "LLM_SAS_LLM_ENDPOINT is not set"}

    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    models_url = base_url + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests_module.get(models_url, headers=headers, timeout=5)
        preview = response.text[:200].replace("\n", " ")
        return {
            "ok": response.ok,
            "status_code": response.status_code,
            "model": model,
            "url": models_url,
            "detail": preview,
        }
    except Exception as exc:
        return {"ok": False, "url": models_url, "model": model, "detail": f"{type(exc).__name__}: {exc}"}


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    module_rows, requests_module = check_modules()
    dataset_rows = check_datasets(project_root)
    env_rows = check_env()
    endpoint_row = check_endpoint(requests_module) if args.check_endpoint else None

    summary = {
        "project_root": str(project_root),
        "modules": module_rows,
        "datasets": dataset_rows,
        "env": env_rows,
        "endpoint": endpoint_row,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    failures = 0
    failures += sum(0 if row["ok"] else 1 for row in module_rows)
    failures += sum(0 if row["exists"] and row["lp_count"] > 0 else 1 for row in dataset_rows)
    if endpoint_row is not None and not endpoint_row.get("ok"):
        failures += 1

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
