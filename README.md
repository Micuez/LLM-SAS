# LLM-LNS

An LLM-guided Large Neighborhood Search framework for combinatorial optimization and MILP problems. The system combines LLM-based heuristic generation, prompt evolution, typed neighborhood operators, structured decision making, and a lightweight reliability layer.

## Overview

This repository implements an evolutionary heuristic-design framework with two interacting loops:

- a prompt evolution loop that improves how the LLM is asked to design heuristics
- an algorithm evolution loop that improves the generated heuristics themselves

For the MILP problems, the framework now goes beyond plain variable scoring. Instead of asking the LLM to directly pick raw variable indices, the system exposes a richer neighborhood-design interface with:

- MILP structural features
- typed neighborhood operators
- structured JSON operator decisions
- a reliability checker with a simple bandit adjustment layer

## Main Innovations

### 1. MILP structural feature augmentation

Each LNS round is informed by multi-level features:

- **global MILP features**: `num_vars`, `num_constraints`, `binary_ratio`, `integer_ratio`, `density`, `avg_var_degree`, `avg_constr_degree`, objective statistics, RHS statistics, LP gap, incumbent objective, best bound, elapsed time, and no-improvement rounds
- **variable features**: `obj_abs`, `degree`, `lp_value`, `incumbent_value`, `fractionality`, `reduced_cost`, `tight_constr_count`, `historical_flip_freq`, and `historical_improve_score`
- **constraint features**: `slack`, `tightness`, `degree`, `violation`, `dual_value`, and `constraint_type_hint`

This turns neighborhood selection from a raw heuristic guess into a state-aware, structure-aware decision problem.

### 2. Typed Neighborhood Operator Library

The MILP pipeline includes a typed operator library so the LLM can choose high-level neighborhood styles instead of directly manipulating variable IDs:

- `FRAC-LNS`: focuses on highly fractional variables
- `TIGHT-LNS`: focuses on variables around tight constraints
- `OBJ-LNS`: focuses on high-impact objective variables
- `GRAPH-BLOCK-LNS`: releases graph-local blocks from the variable-constraint bipartite graph
- `HISTORY-LNS`: exploits historically useful variables
- `DIVERSITY-LNS`: explores rarely released variables
- `HYBRID-LNS`: combines several operators with weights

The runtime converts operator decisions into executable variable scores.

### 3. Structured LLM decision interface

The LLM is prompted with a round-level structured context instead of free-form text only. Each round includes:

- `problem_summary`
- `solver_state`
- `structural_signals`
- `previous_operator_performance`
- `decision_requirement`

The preferred output is JSON, for example:

```json
{
  "operator": "TIGHT-LNS",
  "free_ratio": 0.12,
  "time_budget": 90,
  "focus": "constraints_with_low_slack",
  "exploration_level": "medium",
  "reason": "The current solution has stagnated and tight constraints are concentrated in a small subset of the bipartite graph."
}
```

This makes the decision process more interpretable, more controllable, and easier to analyze.

### 4. Reliability checker and bandit adjustment

The LLM decision is not executed blindly. Before solving each LNS subproblem, the system applies:

- a **simple bandit controller** that scores operators by improvement-per-second, exploration bonus, and failure rate
- a **reliability checker** that repairs unsafe or ineffective decisions

The checker currently handles:

- too few free variables
- too many free variables
- excessive overlap with the previous neighborhood
- overly scattered neighborhoods in sparse graph structure
- repeated operator failure
- frequent timeout pressure

The final execution flow is:

```text
LLM decision
  -> bandit adjustment
  -> reliability checker
  -> neighborhood construction
  -> MILP subproblem solve
  -> operator reward update
```

## Supported Problems

### MILP problems

- **IS**: Independent Set
- **MIKS**: Maximum Independent K-Set
- **MVC**: Minimum Vertex Cover
- **SC**: Set Cover

### Other optimization problems

- **Online Bin Packing**
- **Traveling Salesman Problem**

## Repository Structure

```text
LLM-SAS/
├─ src/
│  ├─ MILP Problems/
│  │  ├─ milp_problem_eoh_common.py
│  │  ├─ IS_eoh_change_prompt_ACP.py
│  │  ├─ MIKS_eoh_change_prompt_ACP.py
│  │  ├─ MVC_eoh_change_prompt_ACP.py
│  │  └─ SC_eoh_change_prompt_ACP.py
│  └─ ...
├─ results/
└─ README.md
```

The MILP scripts now share a common implementation:

- `milp_problem_eoh_common.py` contains the unified MILP EOH pipeline
- each problem-specific script is a thin wrapper with its own `PROBLEM_CONFIG`

## Core Logic

The MILP part of the repository now follows a shared-core architecture.

### Problem wrappers

Each of the four MILP entry files only defines:

- `problem_code`
- `problem_label`
- `problem_prompt_description`
- `instance_path`
- `exp_output_path`

and then forwards control to:

```python
from milp_problem_eoh_common import main
```

This keeps problem-specific differences lightweight and moves the full search logic into one reusable implementation.

### Shared MILP core

The file `src/MILP Problems/milp_problem_eoh_common.py` is the runtime center of the MILP pipeline. It is responsible for:

- loading and resolving instance paths
- building prompt-evolution and algorithm-evolution settings
- extracting MILP structural features
- collecting solver-state and neighborhood-history signals
- querying the LLM through a remote or local OpenAI-compatible endpoint
- translating operator decisions into executable neighborhoods
- running bandit adjustment and reliability checking
- solving subproblems with Gurobi
- recording decision traces and operator rewards

### Why this refactor matters

Compared with the original per-problem duplicated implementation, this shared-core design provides:

- a single place to add new neighborhood operators
- a single place to evolve the LLM prompt interface
- consistent decision logging across IS, MIKS, MVC, and SC
- lower maintenance cost for future ablations and experiments

## End-to-End Workflow

The overall execution workflow for MILP experiments is:

```text
Problem wrapper
  -> configure problem paths and labels
  -> build default runtime parameters
  -> create EOH/LLM evolution engine
  -> load MILP instances
  -> extract structural features
  -> build round-level decision context
  -> query LLM for structured operator JSON
  -> bandit adjustment
  -> reliability checker
  -> construct neighborhood
  -> Gurobi subproblem solve
  -> reward / history / trace update
  -> continue to next round
```

Inside one LNS round, the logic is:

1. read the incumbent solution, LP relaxation state, and constraint activity
2. compute global, variable-level, and constraint-level structural features
3. summarize search-state signals such as stagnation, fractionality concentration, and exploration ratio
4. package the information into a structured prompt
5. ask the LLM to output an operator decision in JSON form
6. let the bandit layer decide whether the proposed operator should be kept or adjusted
7. let the checker repair unsafe choices such as oversized neighborhoods or excessive overlap
8. translate the final typed operator into released-variable scores
9. solve the resulting subproblem with Gurobi under the assigned time budget
10. update reward statistics, operator history, and decision traces

## Innovation Summary

The main research-side contributions of this refactor are the following four modules.

### 1. Structure-aware MILP representation

The framework no longer relies only on raw coefficients or direct variable IDs. Instead, it exposes:

- global MILP statistics
- variable-level structural signals
- constraint-level activity signals
- round-level runtime state

This gives the LLM a better picture of both the optimization landscape and the current search stage.

### 2. Typed operator selection instead of raw variable output

The LLM is encouraged to choose from a library of neighborhood operators rather than emitting unstructured variable sets. This makes decisions:

- more interpretable
- easier to constrain
- easier to compare in ablation studies
- easier to extend with learned or handcrafted operators later

### 3. Structured JSON decision protocol

The LLM now answers with a machine-readable decision object containing:

- operator type
- free ratio
- time budget
- search focus
- exploration level
- explanation

This decouples the semantic decision from the solver implementation and makes downstream control much more robust.

### 4. Reliability and adaptive control

A lightweight control layer now sits between the LLM and the solver:

- the **bandit** layer adjusts decisions using empirical operator performance
- the **checker** layer enforces legal and practical neighborhoods

This prevents brittle behavior such as always selecting the same operator, opening too many variables, or repeatedly timing out.

## Datasets

For MILP problems, datasets are available [here](https://github.com/thuiar/MILPBench/tree/main/Benchmark%20Datasets).

For online bin packing, the data generation is included in the code.

For TSP problems, datasets are available [here](https://github.com/mastqe/tsplib).

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- `gurobipy`
- `numpy`
- `joblib`
- `requests`

## Quick Start

### 1. Configure environment

The MILP pipeline now reads its runtime configuration from environment variables. This works both for remote APIs and for local OpenAI-compatible endpoints such as vLLM, SGLang, and LM Studio.

Example:

```bash
export LLM_SAS_MILP_DATA_ROOT=/path/to/datasets
export LLM_SAS_LLM_ENDPOINT=http://127.0.0.1:8000
export LLM_SAS_LLM_API_KEY=EMPTY
export LLM_SAS_LLM_MODEL=your_local_model_name
export LLM_SAS_LLM_BACKEND=local_openai_compatible
```

If you use a remote provider, set `LLM_SAS_LLM_ENDPOINT`, `LLM_SAS_LLM_API_KEY`, and `LLM_SAS_LLM_MODEL` accordingly. If you do not set `LLM_SAS_MILP_DATA_ROOT`, the code will look for datasets under the `LLM-SAS/` project root.

Before running experiments, you can do a quick environment check:

```bash
python scripts/smoke_check_milp.py
python scripts/smoke_check_milp.py --check-endpoint
```

### 2. Run a MILP problem

```bash
python src/MILP\ Problems/MIKS_eoh_change_prompt_ACP.py
python src/MILP\ Problems/IS_eoh_change_prompt_ACP.py
python src/MILP\ Problems/MVC_eoh_change_prompt_ACP.py
python src/MILP\ Problems/SC_eoh_change_prompt_ACP.py
```

### 3. What happens during execution

For MILP problems, each run will:

1. load LP instances
2. build MILP structural features
3. initialize typed-operator seed heuristics
4. generate and evolve LLM heuristics
5. make structured operator decisions round by round
6. run bandit adjustment and reliability checking
7. solve the resulting LNS subproblems with Gurobi

## MILP Decision Pipeline

The current MILP neighborhood selection pipeline is:

```text
MILP instance
  -> static features
  -> runtime state features
  -> structured decision context
  -> LLM JSON operator decision
  -> typed operator score construction
  -> bandit adjustment
  -> reliability checking
  -> neighborhood mask
  -> Gurobi subproblem
  -> reward/history update
```

## Decision Trace and Analysis

Each MILP run can write round-level decision traces under `results/traces/`. A trace record stores the full decision chain:

- `llm_decision`
- `bandit_decision`
- `checked_decision`
- released variable statistics
- solver runtime and status
- objective change and reward

This makes the framework suitable not only for optimization experiments, but also for operator-behavior analysis and prompt-debugging analysis.

The summarizer script:

```bash
python scripts/summarize_decision_traces.py --trace-dir results/traces
```

aggregates these raw traces into:

- round-level CSV records
- operator-level summaries
- checker-trigger summaries
- a Markdown operator report

## Results

Experimental results and comparisons can be recorded under [`results/`](results/).

Decision traces can be summarized with:

```bash
python scripts/summarize_decision_traces.py --trace-dir results/traces
```

This produces:

- `results/traces/decision_trace_rounds.csv`
- `results/traces/decision_trace_operator_summary.csv`
- `results/traces/decision_trace_checker_summary.csv`
- `results/traces/operator_report.md`

## Notes

- The MILP implementation now prefers structured operator decisions over raw variable-index decisions.
- Raw score output is still supported for backward compatibility.
- The shared MILP core makes it easier to add new problem types or new operator families later.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Issues and pull requests are welcome, especially for:

- new typed neighborhood operators
- improved decision-context design
- stronger bandit or checker policies
- new MILP benchmark integrations
