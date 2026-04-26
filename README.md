# LLM-SAS

LLM-SAS stands for **LLM-guided Structure-Aware Search**. It is a structure-aware Large Neighborhood Search framework for MILP and combinatorial optimization, with a particular focus on letting an LLM make high-level neighborhood decisions from solver state and MILP structure instead of directly emitting raw variable IDs.

## What This Repository Does

This repository keeps the original evolutionary heuristic-design flavor of the project, but the MILP pipeline is now centered on a new structure-aware search layer:

- extract MILP structure and runtime signals
- ask the LLM for a structured neighborhood decision
- map that decision to a typed operator
- repair unsafe decisions with a reliability checker
- solve the resulting LNS subproblem with Gurobi
- update operator history and decision traces

In short, `LLM-SAS` turns LLM-guided neighborhood search from a free-form prompting setup into a controlled, analyzable, structure-aware decision pipeline.

## Main Idea

The central idea is simple:

1. do not ask the LLM to output arbitrary variable indices directly
2. provide the LLM with structural summaries of the MILP instance and the current solver state
3. let it choose a typed neighborhood strategy in structured JSON
4. use a lightweight adaptive layer to make the final decision executable and reliable

This is why the method is called **Structure-Aware Search**.

## Main Innovations

### 1. Structure-aware MILP features

Each LNS round is described by multi-level features rather than only coefficients and incumbent values.

Global features include:

- `num_vars`
- `num_constraints`
- `binary_ratio`
- `integer_ratio`
- `density`
- `avg_var_degree`
- `avg_constr_degree`
- `obj_coef_mean/std/max`
- `rhs_mean/std`
- `lp_gap`
- `incumbent_obj`
- `best_bound`
- `time_elapsed`
- `no_improve_rounds`

Variable features include:

- `obj_abs`
- `degree`
- `lp_value`
- `incumbent_value`
- `fractionality`
- `reduced_cost`
- `tight_constr_count`
- `historical_flip_freq`
- `historical_improve_score`

Constraint features include:

- `slack`
- `tightness`
- `degree`
- `violation`
- `dual_value`
- `constraint_type_hint`

These features let the LLM reason about bottlenecks, stagnation, objective concentration, tight regions, and exploration state.

### 2. Typed Neighborhood Operator Library

Instead of raw variable-index output, the LLM chooses from a typed operator library:

- `FRAC-LNS`
- `TIGHT-LNS`
- `OBJ-LNS`
- `GRAPH-BLOCK-LNS`
- `HISTORY-LNS`
- `DIVERSITY-LNS`
- `HYBRID-LNS`

Their intended roles are:

- `FRAC-LNS`: release highly fractional variables when LP gap is informative
- `TIGHT-LNS`: release variables around tight constraints when local bottlenecks dominate
- `OBJ-LNS`: focus on high-impact objective variables
- `GRAPH-BLOCK-LNS`: release graph-local blocks for sparse large instances
- `HISTORY-LNS`: exploit historically effective variables
- `DIVERSITY-LNS`: force exploration when the search stalls
- `HYBRID-LNS`: combine several operators with weights

This gives the framework a clean action space that is both interpretable and easy to analyze.

### 3. Structured JSON decision protocol

Each round, the LLM receives five blocks of context:

- `Problem Summary`
- `Solver State`
- `Structural Signals`
- `Previous Operator Performance`
- `Decision Requirement`

The preferred output is JSON only, for example:

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

This makes the LLM decision machine-readable, easier to constrain, and easier to log for later analysis.

### 4. Reliability checker and adaptive bandit control

LLM output is not executed blindly.

Before each neighborhood is solved, the framework applies:

- a **bandit layer** that adjusts operators according to historical reward
- a **reliability checker** that repairs invalid or low-quality neighborhood decisions

The checker handles issues such as:

- too few free variables
- too many free variables
- excessive overlap with the previous neighborhood
- overly scattered variables when graph locality is poor
- repeated operator failure
- frequent timeout pressure

The bandit score follows a simple but useful form:

```text
score_k = avg_improvement_per_second_k
        + beta * exploration_bonus_k
        - gamma * failure_rate_k
```

This gives the method a lightweight exploit/explore correction layer on top of LLM decisions.

## Core MILP Logic

The MILP pipeline has been refactored into a shared-core design.

### Shared core

The main implementation lives in:

- `src/MILP Problems/milp_problem_eoh_common.py`

This file is responsible for:

- path resolution and problem configuration
- LLM backend configuration
- feature extraction
- prompt construction
- typed operator execution
- reliability checking
- bandit update
- Gurobi-based LNS solving
- decision trace logging

### Thin problem wrappers

The four MILP problem files:

- `IS_eoh_change_prompt_ACP.py`
- `MIKS_eoh_change_prompt_ACP.py`
- `MVC_eoh_change_prompt_ACP.py`
- `SC_eoh_change_prompt_ACP.py`

are now thin wrappers that only define:

- problem metadata
- dataset path
- output path

and then call the shared `main()` entry in `milp_problem_eoh_common.py`.

This keeps the business logic in one place and makes future extensions much easier.

## End-to-End Workflow

The runtime workflow is:

```text
Problem wrapper
  -> configure paths and problem metadata
  -> build default runtime parameters
  -> load LP instances
  -> solve root / collect incumbent and LP state
  -> extract MILP structural features
  -> build round-level LLM context
  -> query LLM for JSON operator decision
  -> bandit adjustment
  -> reliability checker
  -> convert typed operator to released-variable scores
  -> solve LNS subproblem with Gurobi
  -> update rewards, history, and traces
  -> continue to next round
```

Inside one LNS round, the decision chain is:

```text
LLM decision
  -> bandit decision
  -> checked decision
  -> neighborhood construction
  -> solver outcome
  -> reward update
```

## Decision Trace and Analysis

Each MILP run can write round-level JSONL traces under:

- `results/traces/`

Each record captures:

- `llm_decision`
- `bandit_decision`
- `checked_decision`
- selected operator
- released variable ratio
- solver runtime and status
- objective improvement
- structural signals
- reward-related information

This makes the framework not only runnable, but also inspectable.

Trace summaries can be generated with:

```bash
python scripts/summarize_decision_traces.py --trace-dir results/traces
```

This produces:

- `decision_trace_rounds.csv`
- `decision_trace_operator_summary.csv`
- `decision_trace_checker_summary.csv`
- `operator_report.md`

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
├─ scripts/
│  ├─ smoke_check_milp.py
│  └─ summarize_decision_traces.py
├─ results/
└─ README.md
```

## Supported Problems

### MILP problems

- `IS`: Independent Set
- `MIKS`: Maximum Independent K-Set
- `MVC`: Minimum Vertex Cover
- `SC`: Set Cover

### Other problems already present in the repository

- Online Bin Packing
- Traveling Salesman Problem

## Environment and Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- `gurobipy`
- `numpy`
- `joblib`
- `requests`

The MILP pipeline reads runtime configuration from environment variables:

```bash
export LLM_SAS_MILP_DATA_ROOT=/path/to/datasets
export LLM_SAS_LLM_ENDPOINT=http://127.0.0.1:8000
export LLM_SAS_LLM_API_KEY=EMPTY
export LLM_SAS_LLM_MODEL=your_local_model_name
export LLM_SAS_LLM_BACKEND=local_openai_compatible
```

This supports remote APIs as well as local OpenAI-compatible endpoints such as:

- vLLM
- SGLang
- LM Studio

Before running experiments, you can check the environment with:

```bash
python scripts/smoke_check_milp.py
python scripts/smoke_check_milp.py --check-endpoint
```

Run a MILP problem with:

```bash
python src/MILP\ Problems/MIKS_eoh_change_prompt_ACP.py
python src/MILP\ Problems/IS_eoh_change_prompt_ACP.py
python src/MILP\ Problems/MVC_eoh_change_prompt_ACP.py
python src/MILP\ Problems/SC_eoh_change_prompt_ACP.py
```

## Why LLM-SAS Is Different

Compared with a standard LLM-guided LNS setup, `LLM-SAS` makes three changes in spirit:

- it gives the LLM structured access to MILP geometry and runtime state
- it constrains LLM behavior to a typed, analyzable operator space
- it inserts an adaptive control layer between language decisions and solver execution

So the method is not just "LLM picks a neighborhood." It is:

**LLM + structure signals + typed action space + reliability control + solver feedback**

## Datasets

For MILP problems, benchmark datasets are available from [MILPBench](https://github.com/thuiar/MILPBench/tree/main/Benchmark%20Datasets).

For TSP problems, datasets are available from [TSPLIB-related resources](https://github.com/mastqe/tsplib).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
