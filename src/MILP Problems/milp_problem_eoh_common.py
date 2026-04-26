import os
import shutil
import copy
import numpy as np
import json
import random
import time
import pickle # Not used, can be removed.
import sys
import types
import re
import warnings
import http.client
from typing import Sequence, Tuple # Collection was not used.
import requests
import ast
from gurobipy import GRB, read, Model
import concurrent.futures
import heapq
from joblib import Parallel, delayed
from pathlib import Path
import traceback

THIS_FILE = Path(__file__).resolve()
MILP_DIR = THIS_FILE.parent
SRC_DIR = MILP_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MILP_DATA_ROOT = os.environ.get("LLM_SAS_MILP_DATA_ROOT")

from selection import prob_rank,equal,roulette_wheel,tournament
from management import pop_greedy,ls_greedy,ls_sa

DEFAULT_PROBLEM_CONFIG = {
    "problem_code": "MIKS",
    "problem_label": "Maximum Independent K-Set",
    "problem_prompt_description": "This corresponds to the Maximum Independent K-Set MILP problem, where all constraints are in the form of LHS <= RHS or other standard MILP forms.",
    "instance_path": "./MIKS_easy_instance/LP",
    "exp_output_path": "./MIKS2_ACP/",
}

ACTIVE_PROBLEM_CONFIG = dict(DEFAULT_PROBLEM_CONFIG)


def configure_problem(problem_config=None):
    global ACTIVE_PROBLEM_CONFIG
    ACTIVE_PROBLEM_CONFIG = dict(DEFAULT_PROBLEM_CONFIG)
    if problem_config:
        ACTIVE_PROBLEM_CONFIG.update(problem_config)
    ACTIVE_PROBLEM_CONFIG["instance_path"] = str(_resolve_project_path(ACTIVE_PROBLEM_CONFIG["instance_path"]))
    ACTIVE_PROBLEM_CONFIG["exp_output_path"] = str(_resolve_project_path(ACTIVE_PROBLEM_CONFIG["exp_output_path"]))
    return ACTIVE_PROBLEM_CONFIG


def _resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    if MILP_DATA_ROOT:
        candidate = (Path(MILP_DATA_ROOT).expanduser() / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def _resolve_llm_backend(backend, endpoint):
    backend = (backend or "auto").strip()
    if backend != "auto":
        return backend
    endpoint = (endpoint or "").strip().lower()
    if endpoint.startswith("http://127.0.0.1") or endpoint.startswith("http://localhost"):
        return "local_openai_compatible"
    if endpoint.startswith("https://127.0.0.1") or endpoint.startswith("https://localhost"):
        return "local_openai_compatible"
    if endpoint.startswith("http://"):
        return "local_openai_compatible"
    return "remote"


# Define necessary classes
class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'eoh'                # Method selected
        self.problem = 'milp_construct'     # Problem to solve
        self.selection = None              # Individual selection method (how individuals are selected from the population for evolution)
        self.management = None             # Population management method

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5  # number of algorithms in each population, default = 10
        self.ec_n_pop = 5 # number of populations, default = 10
        self.ec_operators = None # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
        self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]

        #####################
        ### LLM settings  ###
        #####################
        self.llm_api_endpoint = None # endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # model type for remote LLM, e.g., deepseek-chat
        self.llm_backend = "auto"  # auto, remote, or local_openai_compatible

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # if debug
        self.exp_output_path = ACTIVE_PROBLEM_CONFIG["exp_output_path"]  # default folder for ael outputs
        self.exp_n_proc = 1

        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 5 * 300
        self.prompt_eva_timeout = 30
        self.eva_numba_decorator = False


    def set_parallel(self):
        #################################
        ###  Set the number of threads to the maximum available on the machine  ###
        #################################
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")

    def set_ec(self):
        ###########################################################
        ###  Set population management strategy, parent selection strategy, and evolution strategies with corresponding weights
        ###########################################################
        if self.management == None:
            if self.method in ['ael','eoh']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'

        if self.selection == None:
            self.selection = 'prob_rank'


        if self.ec_operators == None:
            if self.method == 'eoh':
                self.ec_operators  = ['e1','e2','m1','m2']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1, 1, 1]
            elif self.method == 'ael':
                self.ec_operators  = ['crossover','mutation']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1]
            elif self.method == 'ls':
                self.ec_operators  = ['m1']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1]
            elif self.method == 'sa':
                self.ec_operators  = ['m1']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1]

        if self.method in ['ls','sa'] and self.ec_pop_size >1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1. ")

    def set_evaluation(self):
        #################################
        ###  Set population evaluation parameters (problem-based)
        #################################
        if self.problem == 'bp_online':
            self.eva_timeout = 20
            self.eva_numba_decorator  = True
        elif self.problem == 'milp_construct':
            self.eva_timeout = 350 * 5

    def set_paras(self, *args, **kwargs):
        #################################
        ###  Set multi-threading, population strategy, and evaluation
        #################################
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Identify and set parallel
        self.set_parallel()

        # Initialize method and ec settings
        self.set_ec()

        # Initialize evaluation settings
        self.set_evaluation()

#######################################
#######################################
###  Alright! Basic settings are done, let's start with prompt settings.
#######################################
#######################################

def create_folders(results_path):
    #####################################################
    ###  Create result folders and subfolders for history, populations, and best individuals
    #####################################################
    folder_path = os.path.join(results_path, "results")

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Remove the existing folder and its contents
        # shutil.rmtree(folder_path) # This line was commented out in the original

        # Create the main folder "results"
        os.makedirs(folder_path)

    # Create subfolders inside "results"
    subfolders = ["history", "pops", "pops_best", "traces"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

class GetPrompts():
    #####################################################
    ###  Prompt class, defines various prompts and their related returns
    #####################################################
    def __init__(self):
        # Prompt for task description
        self.prompt_task = "Given an initial feasible solution and a current solution to a Mixed-Integer Linear Programming (MILP) problem, with a complete description of the constraints and objective function.\
        We want to improve the current solution using Large Neighborhood Search (LNS). \
        The task can be solved step-by-step by starting from the current solution and iteratively selecting a subset of decision variables to relax and re-optimize. \
        In each step, most decision variables are fixed to their values in the current solution, and only a small subset is allowed to change. \
        You need to score all the decision variables based on the information I give you, and I will choose the decision variables with high scores as neighborhood selection.\
        Besides the raw MILP instance data, I will also provide global MILP structure features, variable-level features, constraint-level features, a neighborhood context that indicates whether the current neighborhood should behave like a TIGHT neighborhood or a REPAIR neighborhood, a structured operator performance history, and a structured round-level decision context.\
        I also provide a Typed Neighborhood Operator Library that contains FRAC-LNS, TIGHT-LNS, OBJ-LNS, GRAPH-BLOCK-LNS, HISTORY-LNS, DIVERSITY-LNS, and HYBRID-LNS. You should prefer returning a structured JSON-like operator decision instead of directly choosing variable indices.\
        To avoid getting stuck in local optima, the choice of the subset can incorporate a degree of randomness.\
        You can also consider the correlation between decision variables, for example, assigning similar scores to variables involved in the same constraint, which often exhibit high correlation. This will help me select decision variables from the same constraint.\
        Of course, I also welcome other interesting strategies that you might suggest."
        # Prompt function name: select neighborhood
        self.prompt_func_name = "select_neighborhood"
        # Prompt function inputs:
        # n, m, k, site, value, constraint, initial_solution, current_solution,
        # objective_coefficient, global_features, variable_features,
        # constraint_features, neighborhood_context, operator_performance,
        # decision_context
        self.prompt_func_inputs = [
            "n",
            "m",
            "k",
            "site",
            "value",
            "constraint",
            "initial_solution",
            "current_solution",
            "objective_coefficient",
            "global_features",
            "variable_features",
            "constraint_features",
            "neighborhood_context",
            "operator_performance",
            "decision_context",
        ]
        # Prompt function output: neighbor_score
        self.prompt_func_outputs = ["neighbor_score"]
        # Description of prompt function inputs and outputs
        problem_prompt_description = ACTIVE_PROBLEM_CONFIG["problem_prompt_description"]
        self.prompt_inout_inf = "'n': Number of decision variables in the problem instance. 'n' is a integer number. \
        'm': Number of constraints in the problem instance. 'm' is a integer number.\
        'k': k[i] indicates the number of decision variables involved in the ith constraint. 'k' is a Numpy array with length m.\
        'site': site[i][j] indicates which decision variable is involved in the jth position of the ith constraint. 'site' is a list of Numpy arrays. The length of the list is m.\
        'value': value[i][j] indicates the coefficient of the jth decision variable in the ith constraint. 'value' is a list of Numpy arrays. The length of the list is m.\
        'constraint': constraint[i] indicates the right-hand side value of the ith constraint. 'constraint' is a Numpy array with length m.\
        'initial_solution': initial_solution[i] indicates the initial value of the i-th decision variable. initial_solution is a Numpy array with length n\
        'current_solution': current_solution[i] indicates the current value of the i-th decision variable. current_solution is a Numpy array with length n.\
        'objective_coefficient': objective_coefficient[i] indicates the objective function coefficient corresponding to the i-th decision variable. objective_coefficient is a Numpy array with length n.\
        'global_features': a Python dictionary of scalar MILP structure and search-state features, including 'num_vars', 'num_constraints', 'binary_ratio', 'integer_ratio', 'density', 'avg_var_degree', 'avg_constr_degree', 'obj_coef_mean', 'obj_coef_std', 'obj_coef_max', 'rhs_mean', 'rhs_std', 'lp_gap', 'incumbent_obj', 'best_bound', 'time_elapsed', and 'no_improve_rounds'.\
        'variable_features': a Python dictionary of numpy arrays with length n, including 'obj_abs', 'degree', 'lp_value', 'incumbent_value', 'fractionality', 'reduced_cost', 'tight_constr_count', 'historical_flip_freq', and 'historical_improve_score'.\
        'constraint_features': a Python dictionary of numpy arrays with length m, including 'slack', 'tightness', 'degree', 'violation', 'dual_value', and 'constraint_type_hint'.\
        'neighborhood_context': a Python dictionary describing the current neighborhood round. It contains 'neighborhood_mode', where 0 means TIGHT and 1 means REPAIR, and may also contain 'round_idx', 'parts', and 'target_size'.\
        'operator_performance': a Python dictionary keyed by typed operator names such as 'FRAC-LNS' and 'TIGHT-LNS'. Each entry contains average improvement, average runtime, success rate, and usage count from previous rounds.\
        'decision_context': a Python dictionary with five blocks: 'problem_summary', 'solver_state', 'structural_signals', 'previous_operator_performance', and 'decision_requirement'.\
        'initial_solution', 'current_solution', and 'objective_coefficient' are numpy arrays with length n. The i-th element of the arrays corresponds to the i-th decision variable. " + \
        problem_prompt_description + "\
        You should prefer returning a structured JSON object or a Python dictionary with keys such as 'operator', 'components', 'free_ratio', 'time_budget', 'focus', 'exploration_level', and 'reason'.\
        If you return a JSON string or a dictionary, the TypedNeighborhoodOperatorLibrary in the runtime will convert it to a neighbor score array.\
        'neighbor_score' is also a numpy array that you may create manually. The i-th element of the arrays corresponds to the i-th decision variable."
        # Other descriptions for the prompt function
        self.prompt_other_inf = "Available typed operators are FRAC-LNS, TIGHT-LNS, OBJ-LNS, GRAPH-BLOCK-LNS, HISTORY-LNS, DIVERSITY-LNS, and HYBRID-LNS. Prefer returning JSON only, for example {'operator': 'TIGHT-LNS', 'free_ratio': 0.12, 'time_budget': 90, 'focus': 'constraints_with_low_slack', 'exploration_level': 'medium', 'reason': 'The current solution has stagnated and tight constraints are concentrated.'}. Use the additional MILP features and operator history to bias the neighborhood toward tight constraints in TIGHT mode and toward difficult or stagnated regions in REPAIR mode."

    def get_task(self):
        #################################
        ###  Get task description
        #################################
        return self.prompt_task

    def get_func_name(self):
        #################################
        ###  Get prompt function name
        #################################
        return self.prompt_func_name

    def get_func_inputs(self):
        #################################
        ###  Get prompt function inputs
        #################################
        return self.prompt_func_inputs

    def get_func_outputs(self):
        #################################
        ###  Get prompt function outputs
        #################################
        return self.prompt_func_outputs

    def get_inout_inf(self):
        #################################
        ###  Get description of prompt function inputs and outputs
        #################################
        return self.prompt_inout_inf

    def get_other_inf(self):
        #################################
        ###  Get other descriptions for the prompt function
        #################################
        return self.prompt_other_inf

#######################################
#######################################
###  Time to create problem instances!
#######################################
#######################################

class GetData():
    ########################################################
    ###  Pass instance count and number of cities, get coordinates and distance matrix for each point in each instance
    ########################################################
    def _infer_constraint_type_hint(self, coeffs, rhs, sense):
        if len(coeffs) == 0:
            return 0

        coeffs = np.asarray(coeffs, dtype=float)
        nonnegative = np.all(coeffs >= -1e-9)
        unit_like = np.all(np.abs(np.abs(coeffs) - 1.0) <= 1e-9)
        rhs_integer_like = abs(rhs - round(rhs)) <= 1e-9

        if nonnegative and unit_like and sense == 2:
            return 1  # set-cover-like
        if nonnegative and sense == 1 and unit_like and rhs_integer_like:
            return 3  # cardinality-like
        if nonnegative and sense == 1:
            return 2  # packing-like
        return 0      # generic

    def generate_instances(self, lp_path): #'./test'
        lp_root = Path(lp_path)
        sample_files = [str(path) for path in lp_root.glob("*.lp")]
        if not sample_files:
            raise FileNotFoundError(
                f"No .lp files found under '{lp_root}'. "
                "Please download or place the MILP benchmark instances in the configured instance_path, "
                "or set the environment variable LLM_SAS_MILP_DATA_ROOT to the dataset root directory."
            )
        instance_data = []
        for f in sample_files: # for each instance, randomly generate
            model = read(f)
            model.setParam('OutputFlag', 0)
            value_to_num = {}
            value_num = 0
            # n: number of decision variables
            # m: number of constraints
            # k[i]: number of decision variables in the i-th constraint
            # site[i][j]: which decision variable is at the j-th position of the i-th constraint
            # value[i][j]: coefficient of the j-th decision variable in the i-th constraint
            # constraint[i]: right-hand side value of the i-th constraint
            # constraint_type[i]: type of the i-th constraint (1 for <=, 2 for >=, 3 for ==)
            # coefficient[i]: coefficient of the i-th decision variable in the objective function
            n = model.NumVars
            m = model.NumConstrs
            k = []
            site = []
            value = []
            constraint = []
            constraint_type = []
            for cnstr in model.getConstrs():
                if(cnstr.Sense == '<'):
                    constraint_type.append(1)
                elif(cnstr.Sense == '>'):
                    constraint_type.append(2)
                else:
                    constraint_type.append(3)

                constraint.append(cnstr.RHS)


                now_site = []
                now_value = []
                row = model.getRow(cnstr)
                k.append(row.size())
                for i in range(row.size()):
                    if(row.getVar(i).VarName not in value_to_num.keys()):
                        value_to_num[row.getVar(i).VarName] = value_num
                        value_num += 1
                    now_site.append(value_to_num[row.getVar(i).VarName])
                    now_value.append(row.getCoeff(i))
                site.append(now_site)
                value.append(now_value)

            coefficient = {}
            lower_bound = {}
            upper_bound = {}
            variable_type = {}
            for val in model.getVars():
                if(val.VarName not in value_to_num.keys()):
                    value_to_num[val.VarName] = value_num
                    value_num += 1
                coefficient[value_to_num[val.VarName]] = val.Obj
                lower_bound[value_to_num[val.VarName]] = val.LB
                upper_bound[value_to_num[val.VarName]] = val.UB
                variable_type[value_to_num[val.VarName]] = val.Vtype

            # 1 for minimization, -1 for maximization
            obj_type = model.ModelSense
            lp_solution = np.zeros(n)
            reduced_cost = np.zeros(n)
            dual_value = np.zeros(m)
            lp_obj = 0.0
            try:
                relaxed_model = model.relax()
                relaxed_model.setParam('OutputFlag', 0)
                relaxed_model.optimize()
                if relaxed_model.SolCount > 0:
                    lp_obj = relaxed_model.ObjVal
                    for val in relaxed_model.getVars():
                        idx = value_to_num.get(val.VarName)
                        if idx is None:
                            continue
                        lp_solution[idx] = val.X
                        try:
                            reduced_cost[idx] = val.RC
                        except Exception:
                            reduced_cost[idx] = 0.0
                    for idx, cnstr in enumerate(relaxed_model.getConstrs()):
                        try:
                            dual_value[idx] = cnstr.Pi
                        except Exception:
                            dual_value[idx] = 0.0
            except Exception:
                lp_solution = np.zeros(n)
                reduced_cost = np.zeros(n)
                dual_value = np.zeros(m)
                lp_obj = 0.0

            model.setObjective(0, GRB.MAXIMIZE)
            model.optimize()
            new_sol = {}
            for val in model.getVars():
                if(val.VarName not in value_to_num.keys()):
                    value_to_num[val.VarName] = value_num
                    value_num += 1
                new_sol[value_to_num[val.VarName]] = val.x

            # Post-processing
            new_site = []
            new_value = []
            new_constraint = np.zeros(m)
            new_constraint_type = np.zeros(m, int)
            for i in range(m):
                new_site.append(np.zeros(k[i], int))
                new_value.append(np.zeros(k[i]))
                for j in range(k[i]):
                    new_site[i][j] = site[i][j]
                    new_value[i][j] = value[i][j]
                new_constraint[i] = constraint[i]
                new_constraint_type[i] = constraint_type[i]

            new_coefficient = np.zeros(n)
            new_lower_bound = np.zeros(n)
            new_upper_bound = np.zeros(n)
            new_variable_type = np.zeros(n, int)
            new_new_sol = np.zeros(n)
            for i in range(n):
                new_coefficient[i] = coefficient[i]
                new_lower_bound[i] = lower_bound[i]
                new_upper_bound[i] = upper_bound[i]
                if(variable_type[i] == 'B'):
                    new_variable_type[i] = 0
                elif(variable_type[i] == 'C'):
                    new_variable_type[i] = 1
                else:
                    new_variable_type[i] = 2
                new_new_sol[i] = new_sol[i]

            var_degree = np.zeros(n)
            for i in range(m):
                for j in range(k[i]):
                    var_degree[new_site[i][j]] += 1

            constraint_degree = np.array(k, dtype=float)
            total_nnz = float(np.sum(constraint_degree))
            density = total_nnz / max(float(n * m), 1.0)

            fractionality = np.zeros(n)
            for i in range(n):
                if new_variable_type[i] != 1:
                    fractionality[i] = abs(lp_solution[i] - np.round(lp_solution[i]))

            constraint_type_hint = np.zeros(m, dtype=int)
            for i in range(m):
                constraint_type_hint[i] = self._infer_constraint_type_hint(
                    new_value[i],
                    new_constraint[i],
                    new_constraint_type[i],
                )

            static_global_features = {
                "problem_type": ACTIVE_PROBLEM_CONFIG["problem_code"],
                "num_vars": float(n),
                "num_constraints": float(m),
                "binary_ratio": float(np.mean(new_variable_type == 0)) if n > 0 else 0.0,
                "integer_ratio": float(np.mean(new_variable_type == 2)) if n > 0 else 0.0,
                "density": float(density),
                "avg_var_degree": float(np.mean(var_degree)) if n > 0 else 0.0,
                "avg_constr_degree": float(np.mean(constraint_degree)) if m > 0 else 0.0,
                "obj_coef_mean": float(np.mean(new_coefficient)) if n > 0 else 0.0,
                "obj_coef_std": float(np.std(new_coefficient)) if n > 0 else 0.0,
                "obj_coef_max": float(np.max(new_coefficient)) if n > 0 else 0.0,
                "rhs_mean": float(np.mean(new_constraint)) if m > 0 else 0.0,
                "rhs_std": float(np.std(new_constraint)) if m > 0 else 0.0,
            }
            static_variable_features = {
                "obj_abs": np.abs(new_coefficient),
                "degree": var_degree,
                "lp_value": lp_solution.copy(),
                "fractionality": fractionality,
                "reduced_cost": reduced_cost.copy(),
            }
            static_constraint_features = {
                "degree": constraint_degree.copy(),
                "dual_value": dual_value.copy(),
                "constraint_type_hint": constraint_type_hint,
            }

            instance_data.append({
                "n": n,
                "m": m,
                "k": k,
                "site": new_site,
                "value": new_value,
                "constraint": new_constraint,
                "constraint_type": new_constraint_type,
                "coefficient": new_coefficient,
                "obj_type": obj_type,
                "lower_bound": new_lower_bound,
                "upper_bound": new_upper_bound,
                "variable_type": new_variable_type,
                "initial_solution": new_new_sol,
                "lp_solution": lp_solution,
                "lp_obj": float(lp_obj),
                "static_global_features": static_global_features,
                "static_variable_features": static_variable_features,
                "static_constraint_features": static_constraint_features,
            })
        return instance_data


class TypedNeighborhoodOperatorLibrary:
    OPERATOR_ALIASES = {
        "FRAC": "FRAC-LNS",
        "FRAC-LNS": "FRAC-LNS",
        "TIGHT": "TIGHT-LNS",
        "TIGHT-LNS": "TIGHT-LNS",
        "OBJ": "OBJ-LNS",
        "OBJ-LNS": "OBJ-LNS",
        "GRAPH-BLOCK": "GRAPH-BLOCK-LNS",
        "GRAPH-BLOCK-LNS": "GRAPH-BLOCK-LNS",
        "HISTORY": "HISTORY-LNS",
        "HISTORY-LNS": "HISTORY-LNS",
        "DIVERSITY": "DIVERSITY-LNS",
        "DIVERSITY-LNS": "DIVERSITY-LNS",
        "HYBRID": "HYBRID-LNS",
        "HYBRID-LNS": "HYBRID-LNS",
    }

    @staticmethod
    def canonical_operator_name(name):
        if not isinstance(name, str):
            return "TIGHT-LNS"
        return TypedNeighborhoodOperatorLibrary.OPERATOR_ALIASES.get(name.strip().upper(), name.strip().upper())

    @staticmethod
    def normalize_score(score):
        score = np.asarray(score, dtype=float).reshape(-1)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        if score.size == 0:
            return score
        min_val = np.min(score)
        max_val = np.max(score)
        if max_val - min_val <= 1e-12:
            return np.zeros_like(score)
        return (score - min_val) / (max_val - min_val)

    @staticmethod
    def infer_default_operator(global_features, neighborhood_context):
        if neighborhood_context.get("neighborhood_mode", 0) == 1:
            if global_features.get("no_improve_rounds", 0.0) >= 3:
                return "DIVERSITY-LNS"
            return "HISTORY-LNS"
        if global_features.get("lp_gap", 0.0) >= 0.1:
            return "FRAC-LNS"
        return "TIGHT-LNS"

    @staticmethod
    def default_free_ratio(n, neighborhood_context, spec):
        if isinstance(spec, dict) and "free_ratio" in spec:
            return float(spec["free_ratio"])
        target_size = int(neighborhood_context.get("target_size", max(1, n // max(int(neighborhood_context.get("parts", 10)), 1))))
        return min(max(target_size / max(n, 1), 1.0 / max(n, 1)), 1.0)

    @staticmethod
    def make_operator(operator_name, **kwargs):
        spec = {"operator": TypedNeighborhoodOperatorLibrary.canonical_operator_name(operator_name)}
        spec.update(kwargs)
        return spec

    @staticmethod
    def frac_lns_score(global_features, variable_features, spec):
        frac = TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["fractionality"])
        lp_shift = TypedNeighborhoodOperatorLibrary.normalize_score(np.abs(variable_features["lp_value"] - variable_features["incumbent_value"]))
        gap_boost = min(max(global_features.get("lp_gap", 0.0), 0.0), 1.0)
        score = (0.75 + 0.25 * gap_boost) * frac + 0.35 * lp_shift
        return TypedNeighborhoodOperatorLibrary.normalize_score(score)

    @staticmethod
    def tight_lns_score(instance_data, variable_features, constraint_features, spec):
        n = instance_data["n"]
        m = instance_data["m"]
        k = instance_data["k"]
        site = instance_data["site"]
        tightness = TypedNeighborhoodOperatorLibrary.normalize_score(constraint_features["tightness"])
        violation = TypedNeighborhoodOperatorLibrary.normalize_score(constraint_features["violation"])
        slack = np.abs(constraint_features["slack"])
        inv_slack = 1.0 / (1.0 + slack)
        score = np.zeros(n)
        for i in range(m):
            weight = 0.5 * tightness[i] + 0.3 * violation[i] + 0.2 * inv_slack[i]
            for j in range(k[i]):
                score[site[i][j]] += weight
        score += 0.1 * TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["tight_constr_count"])
        return TypedNeighborhoodOperatorLibrary.normalize_score(score)

    @staticmethod
    def obj_lns_score(variable_features, spec):
        obj_abs = TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["obj_abs"])
        incumbent_value = np.abs(variable_features["incumbent_value"])
        lp_shift = TypedNeighborhoodOperatorLibrary.normalize_score(np.abs(variable_features["lp_value"] - variable_features["incumbent_value"]))
        reduced_cost = TypedNeighborhoodOperatorLibrary.normalize_score(np.abs(variable_features["reduced_cost"]))
        incumbent_level = TypedNeighborhoodOperatorLibrary.normalize_score(incumbent_value)
        score = 0.45 * obj_abs + 0.3 * lp_shift + 0.15 * incumbent_level + 0.1 * reduced_cost
        return TypedNeighborhoodOperatorLibrary.normalize_score(score)

    @staticmethod
    def _select_graph_block(instance_data, variable_features, constraint_features, neighborhood_context, spec):
        n = instance_data["n"]
        m = instance_data["m"]
        k = instance_data["k"]
        site = instance_data["site"]

        free_ratio = TypedNeighborhoodOperatorLibrary.default_free_ratio(n, neighborhood_context, spec)
        target_size = max(1, min(n, int(np.ceil(free_ratio * n))))

        var_to_constraints = [[] for _ in range(n)]
        for i in range(m):
            for j in range(k[i]):
                var_to_constraints[site[i][j]].append(i)

        seed_score = (
            0.35 * TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["degree"])
            + 0.35 * TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["fractionality"])
            + 0.3 * TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["historical_improve_score"])
        )
        seed = int(np.argmax(seed_score)) if n > 0 else 0

        visited_vars = set([seed])
        active_vars = [seed]
        frontier = [seed]

        while frontier and len(active_vars) < target_size:
            current = frontier.pop(0)
            candidate_pairs = []
            for constr_idx in var_to_constraints[current]:
                constr_weight = constraint_features["tightness"][constr_idx] + constraint_features["violation"][constr_idx]
                for pos in range(k[constr_idx]):
                    neighbor = int(site[constr_idx][pos])
                    if neighbor in visited_vars:
                        continue
                    neighbor_weight = (
                        0.4 * constr_weight
                        + 0.35 * variable_features["degree"][neighbor]
                        + 0.25 * variable_features["fractionality"][neighbor]
                    )
                    candidate_pairs.append((neighbor_weight, neighbor))
            candidate_pairs.sort(reverse=True)
            for _, neighbor in candidate_pairs:
                if neighbor in visited_vars:
                    continue
                visited_vars.add(neighbor)
                active_vars.append(neighbor)
                frontier.append(neighbor)
                if len(active_vars) >= target_size:
                    break

        score = np.zeros(n)
        if active_vars:
            score[np.array(active_vars, dtype=int)] = 1.0
        score += 0.15 * seed_score
        return TypedNeighborhoodOperatorLibrary.normalize_score(score)

    @staticmethod
    def history_lns_score(variable_features, spec):
        improve = TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["historical_improve_score"])
        flip = TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["historical_flip_freq"])
        tight = TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["tight_constr_count"])
        score = 0.55 * improve + 0.2 * flip + 0.25 * tight
        return TypedNeighborhoodOperatorLibrary.normalize_score(score)

    @staticmethod
    def diversity_lns_score(variable_features, global_features, spec):
        release_freq = TypedNeighborhoodOperatorLibrary.normalize_score(variable_features["historical_flip_freq"])
        unexplored = 1.0 - release_freq
        lp_shift = TypedNeighborhoodOperatorLibrary.normalize_score(np.abs(variable_features["lp_value"] - variable_features["incumbent_value"]))
        stagnation = min(max(global_features.get("no_improve_rounds", 0.0) / 5.0, 0.0), 1.0)
        score = (0.7 + 0.2 * stagnation) * unexplored + 0.3 * lp_shift
        return TypedNeighborhoodOperatorLibrary.normalize_score(score)

    @staticmethod
    def hybrid_lns_score(spec, instance_data, global_features, variable_features, constraint_features, neighborhood_context):
        components = spec.get("components", [])
        if not components:
            components = [
                {"name": "TIGHT-LNS", "weight": 0.4},
                {"name": "FRAC-LNS", "weight": 0.4},
                {"name": "DIVERSITY-LNS", "weight": 0.2},
            ]

        total_weight = 0.0
        score = np.zeros(instance_data["n"])
        for component in components:
            name = TypedNeighborhoodOperatorLibrary.canonical_operator_name(component.get("name", "TIGHT-LNS"))
            weight = float(component.get("weight", 1.0))
            child_spec = dict(spec)
            child_spec["operator"] = name
            child_score = TypedNeighborhoodOperatorLibrary.build_score(
                child_spec,
                instance_data,
                global_features,
                variable_features,
                constraint_features,
                neighborhood_context,
            )
            score += weight * TypedNeighborhoodOperatorLibrary.normalize_score(child_score)
            total_weight += weight

        if total_weight > 0:
            score = score / total_weight
        return TypedNeighborhoodOperatorLibrary.normalize_score(score)

    @staticmethod
    def build_score(spec, instance_data, global_features, variable_features, constraint_features, neighborhood_context):
        if not isinstance(spec, dict):
            raise TypeError("Operator spec must be a dictionary.")

        operator_name = TypedNeighborhoodOperatorLibrary.canonical_operator_name(
            spec.get("operator", TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context))
        )

        if operator_name == "FRAC-LNS":
            return TypedNeighborhoodOperatorLibrary.frac_lns_score(global_features, variable_features, spec)
        if operator_name == "TIGHT-LNS":
            return TypedNeighborhoodOperatorLibrary.tight_lns_score(instance_data, variable_features, constraint_features, spec)
        if operator_name == "OBJ-LNS":
            return TypedNeighborhoodOperatorLibrary.obj_lns_score(variable_features, spec)
        if operator_name == "GRAPH-BLOCK-LNS":
            return TypedNeighborhoodOperatorLibrary._select_graph_block(instance_data, variable_features, constraint_features, neighborhood_context, spec)
        if operator_name == "HISTORY-LNS":
            return TypedNeighborhoodOperatorLibrary.history_lns_score(variable_features, spec)
        if operator_name == "DIVERSITY-LNS":
            return TypedNeighborhoodOperatorLibrary.diversity_lns_score(variable_features, global_features, spec)
        if operator_name == "HYBRID-LNS":
            return TypedNeighborhoodOperatorLibrary.hybrid_lns_score(
                spec,
                instance_data,
                global_features,
                variable_features,
                constraint_features,
                neighborhood_context,
            )
        return TypedNeighborhoodOperatorLibrary.tight_lns_score(instance_data, variable_features, constraint_features, spec)


class SimpleOperatorBandit:
    def __init__(self, beta=0.15, gamma=0.2):
        self.beta = beta
        self.gamma = gamma

    def _operator_score(self, operator_name, operator_performance, total_rounds):
        perf = operator_performance.get(operator_name, {})
        count = max(int(perf.get("count", 0)), 0)
        avg_improvement = float(perf.get("avg_improvement", 0.0))
        avg_runtime = max(float(perf.get("avg_runtime", 0.0)), 1e-6) if count > 0 else 1.0
        success_rate = float(perf.get("success_rate", 0.0)) if count > 0 else 0.0
        failure_rate = 1.0 - success_rate if count > 0 else 0.0
        exploration_bonus = np.sqrt(np.log(total_rounds + 2.0) / (count + 1.0))
        return avg_improvement / avg_runtime + self.beta * exploration_bonus - self.gamma * failure_rate

    def adjust(self, initial_spec, operator_performance, global_features, neighborhood_context):
        if not isinstance(initial_spec, dict):
            initial_spec = {"operator": TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context)}
        spec = dict(initial_spec)
        operator_name = TypedNeighborhoodOperatorLibrary.canonical_operator_name(
            spec.get("operator", TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context))
        )

        total_rounds = sum(max(int(perf.get("count", 0)), 0) for perf in operator_performance.values())
        candidate_names = list(operator_performance.keys()) or [operator_name]
        candidate_scores = {
            name: self._operator_score(name, operator_performance, total_rounds)
            for name in candidate_names
        }
        best_name = max(candidate_scores, key=candidate_scores.get)
        chosen_score = candidate_scores.get(operator_name, candidate_scores.get(best_name, 0.0))
        best_score = candidate_scores.get(best_name, chosen_score)
        chosen_count = int(operator_performance.get(operator_name, {}).get("count", 0))
        best_count = int(operator_performance.get(best_name, {}).get("count", 0))

        if total_rounds == 0:
            spec["bandit_override"] = False
        elif best_name != operator_name and best_count > 0 and best_score > chosen_score + 1e-6:
            spec["operator"] = best_name
            operator_name = best_name
            spec["bandit_override"] = True
        else:
            spec["bandit_override"] = False

        if operator_performance.get(operator_name, {}).get("success_rate", 0.0) < 0.25 and operator_performance.get(operator_name, {}).get("count", 0) >= 2:
            spec["time_budget"] = min(float(spec.get("time_budget", 120)), 60.0)
            spec["free_ratio"] = min(float(spec.get("free_ratio", 0.15)), 0.12 if neighborhood_context.get("neighborhood_mode", 0) == 0 else 0.18)

        spec["bandit_score"] = float(candidate_scores.get(operator_name, 0.0))
        spec["bandit_ranking"] = candidate_scores
        return spec


class NeighborhoodReliabilityChecker:
    def __init__(self, min_free_ratio=0.05, max_free_ratio=0.3, overlap_threshold=0.8, timeout_threshold=0.35, locality_threshold=0.15):
        self.min_free_ratio = min_free_ratio
        self.max_free_ratio = max_free_ratio
        self.overlap_threshold = overlap_threshold
        self.timeout_threshold = timeout_threshold
        self.locality_threshold = locality_threshold

    def _estimate_overlap_ratio(self, candidate_score, last_released_mask):
        if last_released_mask is None:
            return 0.0
        score = np.asarray(candidate_score, dtype=float).reshape(-1)
        if score.size == 0:
            return 0.0
        target_size = max(1, int(np.sum(last_released_mask)))
        idx = np.argsort(score)[::-1][:target_size]
        new_mask = np.zeros(score.size, dtype=bool)
        new_mask[idx] = True
        denom = max(int(np.sum(last_released_mask)), 1)
        return float(np.sum(new_mask & last_released_mask.astype(bool)) / denom)

    def _estimate_graph_locality(self, candidate_score, instance_data, target_size):
        n = instance_data["n"]
        m = instance_data["m"]
        k = instance_data["k"]
        site = instance_data["site"]
        if n == 0 or m == 0:
            return 1.0
        score = np.asarray(candidate_score, dtype=float).reshape(-1)
        target_size = max(1, min(n, int(target_size)))
        idx = np.argsort(score)[::-1][:target_size]
        selected = set(int(i) for i in idx)
        shared_edges = 0
        possible_edges = 0
        for i in range(m):
            vars_in_constr = [int(site[i][j]) for j in range(k[i]) if int(site[i][j]) in selected]
            c = len(vars_in_constr)
            if c >= 2:
                shared_edges += c * (c - 1) / 2
            possible_edges += c
        if possible_edges <= 1:
            return 0.0
        return float(shared_edges / max(possible_edges, 1.0))

    def adjust(
        self,
        spec,
        instance_data,
        global_features,
        variable_features,
        constraint_features,
        neighborhood_context,
        operator_performance,
        last_released_mask,
        last_operator_name,
        runtime_control,
    ):
        n = instance_data["n"]
        repaired = dict(spec)
        operator_name = TypedNeighborhoodOperatorLibrary.canonical_operator_name(
            repaired.get("operator", TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context))
        )
        repaired["operator"] = operator_name

        free_ratio = float(repaired.get("free_ratio", TypedNeighborhoodOperatorLibrary.default_free_ratio(n, neighborhood_context, repaired)))
        free_ratio = min(max(free_ratio, self.min_free_ratio), self.max_free_ratio)

        perf = operator_performance.get(operator_name, {})
        if perf.get("success_rate", 0.0) < 0.25 and perf.get("count", 0) >= 2:
            repaired["time_budget"] = min(float(repaired.get("time_budget", 120)), 60.0)

        recent_timeout_ratio = float(runtime_control.get("solver_timeout_count", 0) / max(runtime_control.get("round_count", 1), 1))
        if perf.get("timeout_rate", 0.0) > self.timeout_threshold or recent_timeout_ratio > self.timeout_threshold:
            free_ratio = min(free_ratio, 0.12 if neighborhood_context.get("neighborhood_mode", 0) == 0 else 0.18)
            repaired["checker_reason"] = "timeout_shrink"

        probe_spec = dict(repaired)
        probe_spec["free_ratio"] = free_ratio
        probe_score = TypedNeighborhoodOperatorLibrary.build_score(
            probe_spec,
            instance_data,
            global_features,
            variable_features,
            constraint_features,
            neighborhood_context,
        )
        overlap_ratio = self._estimate_overlap_ratio(probe_score, last_released_mask)
        target_size = max(1, int(np.ceil(free_ratio * n)))
        graph_locality = self._estimate_graph_locality(probe_score, instance_data, target_size)
        repaired["overlap_ratio"] = overlap_ratio
        repaired["graph_locality"] = graph_locality

        if overlap_ratio >= self.overlap_threshold:
            if operator_name == "HYBRID-LNS":
                components = list(repaired.get("components", []))
                has_diversity = any(
                    TypedNeighborhoodOperatorLibrary.canonical_operator_name(comp.get("name", "")) == "DIVERSITY-LNS"
                    for comp in components
                )
                if not has_diversity:
                    components.append({"name": "DIVERSITY-LNS", "weight": 0.25})
                repaired["components"] = components
            else:
                repaired = {
                    "operator": "HYBRID-LNS",
                    "components": [
                        {"name": operator_name, "weight": 0.65},
                        {"name": "DIVERSITY-LNS", "weight": 0.35},
                    ],
                    "free_ratio": free_ratio,
                    "time_budget": repaired.get("time_budget", 120),
                    "checker_reason": "forced_diversity_due_to_high_overlap",
                }
                operator_name = "HYBRID-LNS"

        if graph_locality < self.locality_threshold and operator_name not in ["GRAPH-BLOCK-LNS", "HYBRID-LNS"]:
            repaired["operator"] = "GRAPH-BLOCK-LNS"
            operator_name = "GRAPH-BLOCK-LNS"
            free_ratio = min(free_ratio, 0.18)
            repaired["checker_reason"] = "graph_block_compaction"

        if last_operator_name == operator_name and perf.get("success_rate", 0.0) < 0.2 and perf.get("count", 0) >= 3:
            repaired["operator"] = "DIVERSITY-LNS"
            operator_name = "DIVERSITY-LNS"
            free_ratio = max(free_ratio, 0.12)
            repaired["checker_reason"] = "operator_repetition_break"

        repaired["free_ratio"] = min(max(free_ratio, self.min_free_ratio), self.max_free_ratio)
        repaired["time_budget"] = float(repaired.get("time_budget", 120))
        return repaired

class PROBLEMCONST():
    ###########################################
    ###  Initialize MILP problem instances
    ###########################################
    def __init__(self) -> None:
        self.path = ACTIVE_PROBLEM_CONFIG["instance_path"]
        self.set_time = 100
        self.n_p = 5 # test time
        self.epsilon = 1e-3
        self.trace_dir = _resolve_project_path(ACTIVE_PROBLEM_CONFIG["exp_output_path"]) / "results" / "traces"

        self.prompts = GetPrompts()
        # Call defined GetData() to get randomly generated problem instances
        getData = GetData()
        self.instance_data = getData.generate_instances(self.path)
        #print(self.instance_data[0])

    def _trace_file_for_instance(self, instance_data, instance_idx):
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        problem_code = ACTIVE_PROBLEM_CONFIG["problem_code"]
        return self.trace_dir / f"{problem_code.lower()}_decision_trace_instance_{instance_idx}.jsonl"

    def _json_ready(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_ready(v) for v in value]
        return value

    def _write_decision_trace(self, trace_path, trace_record):
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._json_ready(trace_record), ensure_ascii=False) + "\n")

    def _compute_constraint_lhs(self, m, k, site, value, solution):
        lhs = np.zeros(m)
        for i in range(m):
            lhs_i = 0.0
            for j in range(k[i]):
                lhs_i += value[i][j] * solution[site[i][j]]
            lhs[i] = lhs_i
        return lhs

    def _tight_threshold(self, rhs_value):
        return max(self.epsilon, 1e-4 * (1.0 + abs(rhs_value)))

    def _compute_constraint_runtime_features(self, instance_data, current_solution):
        m = instance_data["m"]
        k = instance_data["k"]
        site = instance_data["site"]
        value = instance_data["value"]
        constraint = instance_data["constraint"]
        constraint_type = instance_data["constraint_type"]

        lhs = self._compute_constraint_lhs(m, k, site, value, current_solution)
        slack = np.zeros(m)
        violation = np.zeros(m)
        tightness = np.zeros(m)

        for i in range(m):
            if constraint_type[i] == 1:
                slack[i] = constraint[i] - lhs[i]
                violation[i] = max(lhs[i] - constraint[i], 0.0)
            elif constraint_type[i] == 2:
                slack[i] = lhs[i] - constraint[i]
                violation[i] = max(constraint[i] - lhs[i], 0.0)
            else:
                slack[i] = abs(lhs[i] - constraint[i])
                violation[i] = abs(lhs[i] - constraint[i])

            tightness[i] = 1.0 / (1.0 + abs(slack[i]))
            if violation[i] > self.epsilon:
                tightness[i] = 1.0

        runtime_features = {
            "slack": slack,
            "tightness": tightness,
            "degree": instance_data["static_constraint_features"]["degree"].copy(),
            "violation": violation,
            "dual_value": instance_data["static_constraint_features"]["dual_value"].copy(),
            "constraint_type_hint": instance_data["static_constraint_features"]["constraint_type_hint"].copy(),
            "lhs": lhs,
        }
        return runtime_features

    def _compute_variable_runtime_features(self, instance_data, current_solution, history, constraint_runtime_features):
        n = instance_data["n"]
        m = instance_data["m"]
        k = instance_data["k"]
        site = instance_data["site"]

        tight_constr_count = np.zeros(n)
        for i in range(m):
            is_tight = (
                constraint_runtime_features["violation"][i] > self.epsilon
                or abs(constraint_runtime_features["slack"][i]) <= self._tight_threshold(instance_data["constraint"][i])
            )
            if not is_tight:
                continue
            for j in range(k[i]):
                tight_constr_count[site[i][j]] += 1

        rounds = max(history["round_count"], 1)
        variable_features = {
            "obj_abs": instance_data["static_variable_features"]["obj_abs"].copy(),
            "degree": instance_data["static_variable_features"]["degree"].copy(),
            "lp_value": instance_data["static_variable_features"]["lp_value"].copy(),
            "incumbent_value": current_solution.copy(),
            "fractionality": instance_data["static_variable_features"]["fractionality"].copy(),
            "reduced_cost": instance_data["static_variable_features"]["reduced_cost"].copy(),
            "tight_constr_count": tight_constr_count,
            "historical_flip_freq": history["flip_count"] / rounds,
            "historical_improve_score": history["improve_sum"] / np.maximum(history["release_count"], 1.0),
        }
        return variable_features

    def _compute_lp_gap(self, incumbent_obj, best_bound, obj_type):
        denom = max(abs(incumbent_obj), 1e-6)
        if obj_type == -1:
            return max(best_bound - incumbent_obj, 0.0) / denom
        return max(incumbent_obj - best_bound, 0.0) / denom

    def _build_global_features(self, instance_data, incumbent_obj, best_bound, time_elapsed, no_improve_rounds):
        global_features = dict(instance_data["static_global_features"])
        global_features["lp_gap"] = float(self._compute_lp_gap(incumbent_obj, best_bound, instance_data["obj_type"]))
        global_features["incumbent_obj"] = float(incumbent_obj)
        global_features["best_bound"] = float(best_bound)
        global_features["time_elapsed"] = float(time_elapsed)
        global_features["no_improve_rounds"] = float(no_improve_rounds)
        return global_features

    def _determine_neighborhood_mode(self, constraint_runtime_features, no_improve_rounds):
        if np.max(constraint_runtime_features["violation"]) > self.epsilon:
            return 1
        if no_improve_rounds >= 2:
            return 1
        return 0

    def _concentration_ratio(self, values, top_ratio=0.1):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return 0.0
        arr = np.abs(arr)
        total = float(np.sum(arr))
        if total <= 1e-12:
            return 0.0
        top_k = max(1, int(np.ceil(arr.size * top_ratio)))
        top_vals = np.sort(arr)[-top_k:]
        return float(np.sum(top_vals) / total)

    def _graph_block_modularity_signal(self, instance_data):
        n = instance_data["n"]
        m = instance_data["m"]
        if n <= 0 or m <= 0:
            return 0.0
        density = instance_data["static_global_features"].get("density", 0.0)
        avg_var_degree = instance_data["static_global_features"].get("avg_var_degree", 0.0)
        avg_constr_degree = instance_data["static_global_features"].get("avg_constr_degree", 0.0)
        sparsity = 1.0 - min(max(density, 0.0), 1.0)
        localness = 1.0 / (1.0 + avg_var_degree + avg_constr_degree / max(n, 1))
        return float(min(max(0.65 * sparsity + 0.35 * localness, 0.0), 1.0))

    def _initialize_operator_stats(self):
        operator_names = [
            "FRAC-LNS",
            "TIGHT-LNS",
            "OBJ-LNS",
            "GRAPH-BLOCK-LNS",
            "HISTORY-LNS",
            "DIVERSITY-LNS",
            "HYBRID-LNS",
        ]
        stats = {}
        for name in operator_names:
            stats[name] = {
                "count": 0,
                "success_count": 0,
                "improvement_sum": 0.0,
                "runtime_sum": 0.0,
                "timeout_count": 0,
            }
        return stats

    def _initialize_runtime_control(self):
        return {
            "last_released_mask": None,
            "last_operator_name": None,
            "solver_timeout_count": 0,
            "round_count": 0,
        }

    def _build_operator_performance_summary(self, operator_stats):
        summary = {}
        for name, stat in operator_stats.items():
            raw_count = int(stat["count"])
            count = max(raw_count, 1)
            timeout_rate = float(stat.get("timeout_count", 0) / count) if raw_count > 0 else 0.0
            success_rate = float(stat["success_count"] / count) if raw_count > 0 else 0.0
            summary[name] = {
                "avg_improvement": float(stat["improvement_sum"] / count),
                "avg_runtime": float(stat["runtime_sum"] / count),
                "success_rate": success_rate,
                "failure_rate": float(1.0 - success_rate) if raw_count > 0 else 0.0,
                "timeout_rate": timeout_rate,
                "count": raw_count,
            }
        return summary

    def _update_operator_stats(self, operator_stats, operator_name, improvement, runtime_used, success, timeout_hit=False):
        name = TypedNeighborhoodOperatorLibrary.canonical_operator_name(operator_name)
        if name not in operator_stats:
            operator_stats[name] = {
                "count": 0,
                "success_count": 0,
                "improvement_sum": 0.0,
                "runtime_sum": 0.0,
                "timeout_count": 0,
            }
        operator_stats[name]["count"] += 1
        operator_stats[name]["runtime_sum"] += float(runtime_used)
        operator_stats[name]["improvement_sum"] += max(float(improvement), 0.0)
        if success:
            operator_stats[name]["success_count"] += 1
        if timeout_hit:
            operator_stats[name]["timeout_count"] += 1

    def _get_operator_name_from_selection(self, select_result, global_features, neighborhood_context):
        if isinstance(select_result, str):
            try:
                select_result = json.loads(select_result)
            except Exception:
                return TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context)
        if isinstance(select_result, dict):
            return TypedNeighborhoodOperatorLibrary.canonical_operator_name(
                select_result.get("operator", TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context))
            )
        return TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context)

    def _build_structural_signals(self, instance_data, variable_features, constraint_features, history):
        recent_explored_ratio = float(np.mean(history["release_count"] > 0)) if history["release_count"].size > 0 else 0.0
        signals = {
            "fractionality_concentration": self._concentration_ratio(variable_features["fractionality"]),
            "tight_constraint_ratio": float(np.mean(constraint_features["tightness"] >= 0.8)) if constraint_features["tightness"].size > 0 else 0.0,
            "high_objective_variable_concentration": self._concentration_ratio(variable_features["obj_abs"]),
            "graph_block_modularity": self._graph_block_modularity_signal(instance_data),
            "recent_explored_variable_ratio": recent_explored_ratio,
        }
        return signals

    def _build_decision_context(
        self,
        instance_data,
        global_features,
        structural_signals,
        operator_performance,
        neighborhood_context,
        last_improvement,
        runtime_control,
    ):
        mode_name = "REPAIR" if neighborhood_context.get("neighborhood_mode", 0) == 1 else "TIGHT"
        return {
            "problem_summary": {
                "problem_type": instance_data["static_global_features"].get("problem_type", ACTIVE_PROBLEM_CONFIG["problem_code"]),
                "num_vars": int(instance_data["n"]),
                "num_constraints": int(instance_data["m"]),
                "density": float(global_features.get("density", 0.0)),
                "binary_ratio": float(global_features.get("binary_ratio", 0.0)),
                "avg_var_degree": float(global_features.get("avg_var_degree", 0.0)),
                "avg_constr_degree": float(global_features.get("avg_constr_degree", 0.0)),
            },
            "solver_state": {
                "incumbent_objective": float(global_features.get("incumbent_obj", 0.0)),
                "best_bound": float(global_features.get("best_bound", 0.0)),
                "current_gap": float(global_features.get("lp_gap", 0.0)),
                "no_improvement_rounds": float(global_features.get("no_improve_rounds", 0.0)),
                "last_improvement": float(last_improvement),
                "elapsed_time": float(global_features.get("time_elapsed", 0.0)),
            },
            "structural_signals": structural_signals,
            "previous_operator_performance": operator_performance,
            "decision_requirement": {
                "required_fields": ["operator", "free_ratio", "time_budget", "focus", "exploration_level", "reason"],
                "preferred_output": "json_only",
                "neighborhood_mode": mode_name,
                "target_free_ratio": float(max(1, neighborhood_context.get("target_size", 1)) / max(instance_data["n"], 1)),
                "recent_timeout_ratio": float(runtime_control["solver_timeout_count"] / max(runtime_control["round_count"], 1)),
            },
        }

    def _normalize_llm_decision(self, select_result, global_features, neighborhood_context):
        if not isinstance(select_result, (str, dict)):
            return select_result
        if isinstance(select_result, str):
            try:
                select_result = json.loads(select_result)
            except Exception:
                select_result = {"operator": TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context)}
        if isinstance(select_result, dict):
            normalized = dict(select_result)
            normalized["operator"] = TypedNeighborhoodOperatorLibrary.canonical_operator_name(
                normalized.get("operator", TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context))
            )
            return normalized
        return {
            "operator": TypedNeighborhoodOperatorLibrary.infer_default_operator(global_features, neighborhood_context),
            "free_ratio": TypedNeighborhoodOperatorLibrary.default_free_ratio(1, neighborhood_context, {}),
            "reason": "fallback_from_raw_score",
        }

    def _build_neighborhood_mask(self, base_score, instance_data, constraint_runtime_features, mode, parts):
        n = instance_data["n"]
        m = instance_data["m"]
        k = instance_data["k"]
        site = instance_data["site"]

        score = np.asarray(base_score, dtype=float).reshape(-1)
        if score.shape[0] != n:
            raise ValueError(f"select_neighborhood must return an array of length {n}, got {score.shape[0]}.")
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        support = np.zeros(n)
        for i in range(m):
            if mode == 1:
                weight = constraint_runtime_features["violation"][i] + constraint_runtime_features["tightness"][i]
            else:
                weight = constraint_runtime_features["tightness"][i]
            if weight <= 0:
                continue
            for j in range(k[i]):
                support[site[i][j]] += weight

        max_support = np.max(np.abs(support))
        if max_support > 0:
            support = support / max_support

        combined_score = score + (0.5 if mode == 1 else 0.25) * support

        base_target = max(1, n // max(parts, 1))
        target_size = base_target if mode == 0 else min(n, max(base_target + max(1, base_target // 2), 1))
        indices = np.argsort(combined_score)[::-1]
        color = np.zeros(n)
        color[indices[:target_size]] = 1
        return color, target_size

    def _resolve_neighbor_score(
        self,
        select_result,
        instance_data,
        global_features,
        variable_features,
        constraint_features,
        neighborhood_context,
        decision_context,
    ):
        if isinstance(select_result, str):
            try:
                select_result = json.loads(select_result)
            except Exception:
                pass
        if isinstance(select_result, dict):
            return TypedNeighborhoodOperatorLibrary.build_score(
                select_result,
                instance_data,
                global_features,
                variable_features,
                constraint_features,
                neighborhood_context,
            )
        return np.asarray(select_result, dtype=float)

    def _objective_improvement(self, previous_obj, new_obj, obj_type):
        if obj_type == -1:
            return new_obj - previous_obj
        return previous_obj - new_obj

    def _update_history(self, history, released_mask, previous_solution, new_solution, improvement):
        released = released_mask.astype(bool)
        changed = released & (np.abs(new_solution - previous_solution) > self.epsilon)
        history["round_count"] += 1
        history["release_count"][released] += 1.0
        history["flip_count"][changed] += 1.0
        if improvement > 0:
            history["improve_sum"][released] += improvement

    def _initialize_history(self, n):
        return {
            "round_count": 0,
            "release_count": np.zeros(n),
            "flip_count": np.zeros(n),
            "improve_sum": np.zeros(n),
        }

    def Gurobi_solver(self, n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, variable_type, now_sol, now_col):
        '''
        Function Description:
        Solves the given problem instance using Gurobi solver.

        Parameter Description:
        - n: Number of decision variables in the problem instance.
        - m: Number of constraints in the problem instance.
        - k: k[i] indicates the number of decision variables in the i-th constraint.
        - site: site[i][j] indicates which decision variable is at the j-th position of the i-th constraint.
        - value: value[i][j] indicates the coefficient of the j-th decision variable in the i-th constraint.
        - constraint: constraint[i] indicates the right-hand side value of the i-th constraint.
        - constraint_type: constraint_type[i] indicates the type of the i-th constraint (1 for <=, 2 for >=).
        - coefficient: coefficient[i] indicates the coefficient of the i-th decision variable in the objective function.
        - time_limit: Maximum solving time.
        - obj_type: Whether the problem is maximization or minimization.
        '''
        # Get start time
        begin_time = time.time()
        # Define the optimization model
        model = Model("Gurobi")
        # Set variable mapping
        site_to_new = {}
        new_to_site = {}
        new_num = 0
        x = []
        for i in range(n):
            if(now_col[i] == 1):
                site_to_new[i] = new_num
                new_to_site[new_num] = i
                new_num += 1
                if(variable_type[i] == 0):
                    x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
                elif(variable_type[i] == 1):
                    x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
                else:
                    x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))

        # Set objective function and optimization goal (maximize/minimize)
        coeff = 0
        for i in range(n):
            if(now_col[i] == 1):
                coeff += x[site_to_new[i]] * coefficient[i]
            else:
                coeff += now_sol[i] * coefficient[i]
        if(obj_type == -1):
            model.setObjective(coeff, GRB.MAXIMIZE)
        else:
            model.setObjective(coeff, GRB.MINIMIZE)
        # Add m constraints
        for i in range(m):
            constr = 0
            flag = 0
            for j in range(k[i]):
                if(now_col[site[i][j]] == 1):
                    constr += x[site_to_new[site[i][j]]] * value[i][j]
                    flag = 1
                else:
                    constr += now_sol[site[i][j]] * value[i][j]

            if(flag == 1):
                if(constraint_type[i] == 1):
                    model.addConstr(constr <= constraint[i])
                elif(constraint_type[i] == 2):
                    model.addConstr(constr >= constraint[i])
                else:
                    model.addConstr(constr == constraint[i])
            else:
                if(constraint_type[i] == 1):
                    if(constr > constraint[i]):
                        pass # Original code had print statements for debug
                else:
                    if(constr < constraint[i]):
                        pass # Original code had print statements for debug
        # Set maximum solving time
        model.setParam('OutputFlag', 0)
        if(time_limit - (time.time() - begin_time) <= 0):
            return -1, -1, -1
        model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
        # Optimize
        model.optimize()
        try:
            new_sol = np.zeros(n)
            for i in range(n):
                if(now_col[i] == 0):
                    new_sol[i] = now_sol[i]
                else:
                    if(variable_type[i] == 1):
                        new_sol[i] = x[site_to_new[i]].X
                    else:
                        new_sol[i] = (int)(x[site_to_new[i]].X)

            return new_sol, model.ObjVal, 1
        except:
            return -1, -1, -1

    def eval(self, n, coefficient, new_sol):
        ans = 0
        for i in range(n):
            ans += coefficient[i] * new_sol[i]
        return(ans)

    def greedy_one(self, now_instance_data, eva, instance_idx=0):
        n = now_instance_data["n"]
        m = now_instance_data["m"]
        k = now_instance_data["k"]
        site = now_instance_data["site"]
        value = now_instance_data["value"]
        constraint = now_instance_data["constraint"]
        constraint_type = now_instance_data["constraint_type"]
        coefficient = now_instance_data["coefficient"]
        obj_type = now_instance_data["obj_type"]
        lower_bound = now_instance_data["lower_bound"]
        upper_bound = now_instance_data["upper_bound"]
        variable_type = now_instance_data["variable_type"]
        initial_sol = now_instance_data["initial_solution"]
        best_bound = now_instance_data["lp_obj"]

        parts = 10
        begin_time = time.time()
        trace_path = self._trace_file_for_instance(now_instance_data, instance_idx)
        turn_ans = [self.eval(n, coefficient, initial_sol)]
        history = self._initialize_history(n)
        operator_stats = self._initialize_operator_stats()
        runtime_control = self._initialize_runtime_control()
        bandit = SimpleOperatorBandit()
        checker = NeighborhoodReliabilityChecker()
        no_improve_rounds = 0
        round_idx = 0
        last_improvement = 0.0

        now_sol = initial_sol.copy()
        try:
            while(time.time() - begin_time <= self.set_time):
                current_obj = self.eval(n, coefficient, now_sol)
                constraint_features = self._compute_constraint_runtime_features(now_instance_data, now_sol)
                variable_features = self._compute_variable_runtime_features(now_instance_data, now_sol, history, constraint_features)
                global_features = self._build_global_features(
                    now_instance_data,
                    current_obj,
                    best_bound,
                    time.time() - begin_time,
                    no_improve_rounds,
                )
                neighborhood_mode = self._determine_neighborhood_mode(constraint_features, no_improve_rounds)
                neighborhood_context = {
                    "neighborhood_mode": neighborhood_mode,
                    "round_idx": round_idx,
                    "parts": parts,
                    "target_size": max(1, n // max(parts, 1)),
                }
                structural_signals = self._build_structural_signals(
                    now_instance_data,
                    variable_features,
                    constraint_features,
                    history,
                )
                operator_performance = self._build_operator_performance_summary(operator_stats)
                decision_context = self._build_decision_context(
                    now_instance_data,
                    global_features,
                    structural_signals,
                    operator_performance,
                    neighborhood_context,
                    last_improvement,
                    runtime_control,
                )

                select_result = eva.select_neighborhood(
                                    n,
                                    m,
                                    copy.deepcopy(k),
                                    copy.deepcopy(site),
                                    copy.deepcopy(value),
                                    copy.deepcopy(constraint),
                                    copy.deepcopy(initial_sol),
                                    copy.deepcopy(now_sol),
                                    copy.deepcopy(coefficient),
                                    copy.deepcopy(global_features),
                                    copy.deepcopy(variable_features),
                                    copy.deepcopy(constraint_features),
                                    copy.deepcopy(neighborhood_context),
                                    copy.deepcopy(operator_performance),
                                    copy.deepcopy(decision_context),
                                )
                normalized_decision = self._normalize_llm_decision(
                    select_result,
                    global_features,
                    neighborhood_context,
                )
                llm_decision_for_trace = normalized_decision if isinstance(normalized_decision, dict) else {
                    "raw_score_mode": True,
                    "reason": "heuristic_returned_neighbor_score_directly",
                }
                if isinstance(normalized_decision, dict):
                    bandit_decision = bandit.adjust(
                        normalized_decision,
                        operator_performance,
                        global_features,
                        neighborhood_context,
                    )
                    checked_decision = checker.adjust(
                        bandit_decision,
                        now_instance_data,
                        global_features,
                        variable_features,
                        constraint_features,
                        neighborhood_context,
                        operator_performance,
                        runtime_control["last_released_mask"],
                        runtime_control["last_operator_name"],
                        runtime_control,
                    )
                    selected_operator_name = self._get_operator_name_from_selection(
                        checked_decision,
                        global_features,
                        neighborhood_context,
                    )
                    score_source = checked_decision
                    bandit_decision_for_trace = bandit_decision
                    checked_decision_for_trace = checked_decision
                else:
                    selected_operator_name = TypedNeighborhoodOperatorLibrary.infer_default_operator(
                        global_features,
                        neighborhood_context,
                    )
                    score_source = normalized_decision
                    bandit_decision_for_trace = {"skipped": True, "reason": "raw_score_mode"}
                    checked_decision_for_trace = {"skipped": True, "reason": "raw_score_mode"}
                neighbor_score = self._resolve_neighbor_score(
                    score_source,
                    now_instance_data,
                    global_features,
                    variable_features,
                    constraint_features,
                    neighborhood_context,
                    decision_context,
                )
                color, target_size = self._build_neighborhood_mask(
                    neighbor_score,
                    now_instance_data,
                    constraint_features,
                    neighborhood_mode,
                    parts,
                )
                neighborhood_context["target_size"] = target_size
                if(self.set_time - (time.time() - begin_time) <= 0):
                    break
                previous_sol = now_sol.copy()
                previous_obj = current_obj
                solve_begin = time.time()
                new_sol, now_val, now_flag = self.Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(self.set_time - (time.time() - begin_time), self.set_time / 5), obj_type, lower_bound, upper_bound, variable_type, now_sol, color)
                solve_runtime = time.time() - solve_begin
                runtime_control["round_count"] += 1
                runtime_control["last_released_mask"] = color.astype(bool)
                runtime_control["last_operator_name"] = selected_operator_name
                if(now_flag == -1):
                    runtime_control["solver_timeout_count"] += 1
                    self._update_operator_stats(operator_stats, selected_operator_name, 0.0, solve_runtime, False, timeout_hit=True)
                    trace_record = {
                        "instance_idx": instance_idx,
                        "round_idx": round_idx,
                        "problem_code": ACTIVE_PROBLEM_CONFIG["problem_code"],
                        "llm_decision": llm_decision_for_trace,
                        "bandit_decision": bandit_decision_for_trace,
                        "checked_decision": checked_decision_for_trace,
                        "selected_operator": selected_operator_name,
                        "global_features": global_features,
                        "structural_signals": structural_signals,
                        "operator_performance_before": operator_performance,
                        "decision_context": decision_context,
                        "target_size": int(target_size),
                        "released_count": int(np.sum(color)),
                        "released_ratio": float(np.sum(color) / max(n, 1)),
                        "solver_runtime": float(solve_runtime),
                        "solver_status": "timeout_or_failure",
                        "objective_before": float(previous_obj),
                        "objective_after": None,
                        "improvement": 0.0,
                        "timeout_count": int(runtime_control["solver_timeout_count"]),
                        "elapsed_time": float(time.time() - begin_time),
                    }
                    self._write_decision_trace(trace_path, trace_record)
                    no_improve_rounds += 1
                    last_improvement = 0.0
                    round_idx += 1
                    continue
                now_sol = new_sol
                turn_ans.append(now_val)
                improvement = self._objective_improvement(previous_obj, now_val, obj_type)
                self._update_operator_stats(operator_stats, selected_operator_name, improvement, solve_runtime, improvement > self.epsilon, timeout_hit=False)
                self._update_history(history, color, previous_sol, new_sol, improvement)
                trace_record = {
                    "instance_idx": instance_idx,
                    "round_idx": round_idx,
                    "problem_code": ACTIVE_PROBLEM_CONFIG["problem_code"],
                    "llm_decision": llm_decision_for_trace,
                    "bandit_decision": bandit_decision_for_trace,
                    "checked_decision": checked_decision_for_trace,
                    "selected_operator": selected_operator_name,
                    "global_features": global_features,
                    "structural_signals": structural_signals,
                    "operator_performance_before": operator_performance,
                    "decision_context": decision_context,
                    "target_size": int(target_size),
                    "released_count": int(np.sum(color)),
                    "released_ratio": float(np.sum(color) / max(n, 1)),
                    "solver_runtime": float(solve_runtime),
                    "solver_status": "success",
                    "objective_before": float(previous_obj),
                    "objective_after": float(now_val),
                    "improvement": float(improvement),
                    "timeout_count": int(runtime_control["solver_timeout_count"]),
                    "elapsed_time": float(time.time() - begin_time),
                }
                self._write_decision_trace(trace_path, trace_record)
                last_improvement = improvement
                if improvement > self.epsilon:
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                if(len(turn_ans) > 3 and abs(turn_ans[-1] - turn_ans[-3]) <= self.epsilon * max(abs(turn_ans[-1]), 1.0) and parts >= 3):
                    parts -= 1
                round_idx += 1
            return(-turn_ans[-1])
        except Exception as e:
            print(f"MILP Error: {e}")
            traceback.print_exc()
            return(1e9)

    def run_with_timeout(self, time_limit, func, *args, **kwargs):
        """Run function func within time_limit seconds, return None if timeout"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                # Wait for result, at most time_limit seconds
                return future.result(timeout=time_limit)
            except concurrent.futures.TimeoutError:
                # Timeout, return None
                print(f"Function {func.__name__} timed out after {time_limit} seconds.")
                return None

    def greedy(self, eva):
        ###############################################################################
        ###  Use a greedy-like approach, selecting the next neighborhood in each step via eva's select_neighborhood
        ###  Run multiple instances, returning the average of their results
        ###############################################################################
        results = []
        try:
            num = 0
            for now_instance_data in self.instance_data:
                num += 1
                #print("QWQ!", num)
                result = self.run_with_timeout(150, self.greedy_one, now_instance_data, eva, num)
                #result = self.greedy_one(now_instance_data, eva)

                if result is not None:
                    results.append(result)
                else:
                    results.append(1e9)
                #print("QAQ!", num, result)
                #print("QAQ!")
            # Generate self.pop_size offspring individuals. Results are stored in results list, where each element is a (p, off) tuple, with p being the parent and off being the generated offspring.
            # results = Parallel(n_jobs=self.n_p,timeout=self.set_time+30)(delayed(self.greedy_one)(now_instance_data, eva) for now_instance_data in self.instance_data)
        except Exception as e:
            print(f"Parallel MILP Error: {e}")
            traceback.print_exc()
            results = [1e9]

        return sum(results) / len(results)


    def evaluate(self, code_string):
        ###############################################################################
        ###  Call greedy to evaluate the fitness of the current strategy
        ###  Key question: How to transform the current strategy (string) into executable code?
        ###############################################################################
        try:
            # Use warnings.catch_warnings() to capture and control warnings generated during code execution
            with warnings.catch_warnings():
                # Set captured warnings to ignore mode. This means any warnings generated within this code block will be ignored and not displayed to the user.
                warnings.simplefilter("ignore")

                # Create a new module object named "heuristic_module". types.ModuleType creates a new empty module, similar to a container for the code to be executed.
                heuristic_module = types.ModuleType("heuristic_module")
                heuristic_module.__dict__["TypedNeighborhoodOperatorLibrary"] = TypedNeighborhoodOperatorLibrary

                # Use the exec function to execute code_string within the namespace of the heuristic_module.
                # exec can dynamically execute string-form code and store the results in the specified namespace.
                exec(code_string, heuristic_module.__dict__)

                # Add the newly created module to sys.modules so it can be imported like a normal module. sys.modules is a dictionary that stores all imported modules.
                # By adding heuristic_module to this dictionary, other parts of the code can access it using import heuristic_module.
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Call a method self.greedy within the class, passing heuristic_module as an argument.
                # This line of code will return a fitness value, based on the logic defined in the passed heuristic_module.
                fitness = self.greedy(heuristic_module)
                # If no exception occurs, return the calculated fitness value.
                return fitness
        except Exception as e:
            # Return None in case of an exception, indicating code execution failure or inability to calculate fitness.
            print(f"Greedy MILP Error: {e}")
            return None


class Probs():
    ###########################################
    ###  Load existing problem instances or create new ones using PROBLEMCONST()
    ###########################################
    def __init__(self,paras):
        if not isinstance(paras.problem, str):
            # Load existing problem instances
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "milp_construct":
            # Create new problem instances
            self.prob = PROBLEMCONST()
            print("- Prob "+paras.problem+" loaded ")
        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):
        # Return the problem instance
        return self.prob


#######################################
#######################################
###  Finally, the section for interacting with the large language model!
#######################################
#######################################
class InterfaceAPI:
    #######################################
    ###  Call API to communicate with the large language model
    #######################################
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, backend="auto"):
        self.api_endpoint = api_endpoint # API endpoint address
        self.api_key = api_key           # API key
        self.model_LLM = model_LLM       # Name of the large language model used
        self.debug_mode = debug_mode     # Whether debug mode is enabled
        self.n_trial = 5                 # Indicates a maximum of 5 attempts when trying to get a response
        self.backend = _resolve_llm_backend(backend, api_endpoint)

    def _normalize_base_url(self):
        endpoint = (self.api_endpoint or "").strip()
        if not endpoint:
            return ""
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            base_url = endpoint.rstrip("/")
        else:
            scheme = "http" if self.backend == "local_openai_compatible" else "https"
            base_url = f"{scheme}://{endpoint.rstrip('/')}"
        if base_url.endswith("/v1"):
            return base_url
        return base_url + "/v1"

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key not in [None, "", "EMPTY", "empty", "none", "None"]:
            headers["Authorization"] = "Bearer " + self.api_key
        if self.backend != "local_openai_compatible":
            headers["User-Agent"] = "Apifox/1.0.0 (https://apifox.com)"
            headers["x-api2d-no-cache"] = "1"
        return headers

    def get_response(self, prompt_content):
        payload_explanation = {
            "model": self.model_LLM,
            "messages": [
                {"role": "user", "content": prompt_content}
            ],
        }
        headers = self._headers()
        base_url = self._normalize_base_url()
        request_url = base_url + "/chat/completions"

        response = None
        for n_trial in range(1, self.n_trial + 1):
            try:
                res = requests.post(
                    request_url,
                    headers=headers,
                    json=payload_explanation,
                    timeout=120,
                )
                res.raise_for_status()
                json_data = res.json()
                response = json_data["choices"][0]["message"]["content"]
                return response
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in API trial {n_trial}/{self.n_trial}: {e}")
                continue

        return response


class InterfaceLLM:
    #######################################
    ###  Call InterfaceAPI class to communicate with the large language model
    #######################################
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, backend="auto"):
        self.api_endpoint = api_endpoint # API endpoint URL, used for communication with the language model
        self.api_key = api_key           # API key, used for authentication
        self.model_LLM = model_LLM       # Name of the language model used
        self.debug_mode = debug_mode     # Whether debug mode is enabled
        self.backend = _resolve_llm_backend(backend, api_endpoint)

        print("- check LLM API")

        if self.backend == "local_openai_compatible":
            print("local OpenAI-compatible llm endpoint is used ...")
        elif self.backend == "remote":
            print("remote llm api is used ...")
        else:
            print("auto llm endpoint mode is used ...")

        if self.api_endpoint == None or self.api_endpoint == 'xxx':
            print(">> Stop with wrong API setting: Set llm_api_endpoint to a remote host or local OpenAI-compatible base URL.")
            exit()
        if self.backend != "local_openai_compatible" and (self.api_key == None or self.api_key == 'xxx'):
            print(">> Stop with wrong API setting: Set llm_api_key for remote APIs, or switch llm_backend to local_openai_compatible.")
            exit()
        # Create an instance of InterfaceAPI class and assign it to self.interface_llm. InterfaceAPI is a class defined above for handling actual API requests.
        self.interface_llm = InterfaceAPI(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            self.debug_mode,
            self.backend,
        )

        # Call the get_response method of the InterfaceAPI instance, sending a simple request "1+1=?" to test if the API connection and configuration are correct.
        res = self.interface_llm.get_response("1+1=?")

        # Check if the response is None, which means the API request failed or configuration is incorrect.
        if res == None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()

    def get_response(self, prompt_content):
        ##############################################################################
        # Define a method get_response to get LLM's response to given content.
        # It accepts one parameter prompt_content, representing the prompt content provided by the user.
        ##############################################################################

        # Call the get_response method of the InterfaceAPI instance, send the prompt content and get the response.
        response = self.interface_llm.get_response(prompt_content)

        # Return the response obtained from InterfaceAPI.
        return response


#######################################
#######################################
###  Outer layer! Seems ready? Let's prepare for evolution!
#######################################
#######################################
class Evolution_Prompt():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, problem_type, backend="auto", **kwargs):
        # problem_type:minimization/maximization

        self.prompt_task = "We are working on solving a " + problem_type + " problem." + \
        " Our objective is to leverage the capabilities of the Language Model (LLM) to generate heuristic algorithms that can efficiently tackle this problem." + \
        " We have already developed a set of initial prompts and observed the corresponding outputs." + \
        " However, to improve the effectiveness of these algorithms, we need your assistance in carefully analyzing the existing prompts and their results." + \
        " Based on this analysis, we ask you to generate new prompts that will help us achieve better outcomes in solving the " + problem_type + " problem."

        # LLM settings
        self.api_endpoint = api_endpoint      # LLM API endpoint for interaction with external service.
        self.api_key = api_key                # API key for authentication and authorization.
        self.model_LLM = model_LLM            # Name or identifier of the language model used.
        self.debug_mode = debug_mode          # Debug mode flag
        self.backend = backend

        # Use the defined InterfaceLLM to set up LLM, then use its get_response(self, prompt_content) function for communication.
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode, self.backend)


    def get_prompt_cross(self,prompts_indivs):
        ##################################################
        ###  Generate prompt crossover method
        ##################################################

        # Combine the algorithm ideas and corresponding codes from individuals into statements like "Algorithm No. 1's idea and code are... Algorithm No. 2's idea and code are..."
        prompt_indiv = ""
        for i in range(len(prompts_indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" prompt's tasks assigned to LLM, and objective function value are: \n" + prompts_indivs[i]['prompt']+"\n" + str(prompts_indivs[i]['objective']) +"\n"
        # 1. Describe the task
        # 2. Tell the LLM how many algorithms we are providing and what they are like (combined with prompt_indiv)
        # 3. Make a request, hoping the LLM creates an algorithm completely different from the previously provided ones
        # 4. Tell the LLM to first describe your new algorithm and main steps in one sentence, with the description enclosed in parentheses.
        # 5. Next, implement it in Python as a function named self.prompt_func_name
        # 6. Tell the LLM how many inputs and outputs this function has, and what inputs and outputs (corresponding to processed self.prompt_func_inputs and self.prompt_func_outputs)
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize not to add extra explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(prompts_indivs))+" existing prompt with objective function value as follows: \n"\
+prompt_indiv+\
"Please help me create a new prompt that has a totally different form from the given ones but can be motivated from them. \n" +\
"Please describe your new prompt and main steps in one sentences."\
+"\n"+"Do not give additional explanations!!! Just one sentences." \
+"\n"+"Do not give additional explanations!!! Just one sentences."
        return prompt_content


    def get_prompt_variation(self,prompts_indivs):
        ##################################################
        ###  Prompt for modifying a heuristic to improve performance
        ##################################################

        # 1. Describe the task
        # 2. Tell the LLM that we are providing one algorithm, describe its idea and code
        # 3. Make a request, hoping the LLM creates a new algorithm with a different form but which can be a modified version of the provided algorithm
        # 4. Tell the LLM to first describe your new algorithm and main steps in one sentence, with the description enclosed in parentheses.
        # 5. Next, implement it in Python as a function named self.prompt_func_name
        # 6. Tell the LLM how many inputs and outputs this function has, and what inputs and outputs (corresponding to processed self.prompt_func_inputs and self.prompt_func_outputs)
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize not to add extra explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"I have one prompt with its objective function value as follows." + \
"prompt description: " + prompts_indivs[0]['prompt'] + "\n" + \
"objective function value:\n" +\
str(prompts_indivs[0]['objective'])+"\n" +\
"Please assist me in creating a new prompt that has a different form but can be a modified version of the algorithm provided. \n" +\
"Please describe your new prompt and main steps in one sentences." \
+"\n"+"Do not give additional explanations!!! Just one sentences." \
+"\n"+"Do not give additional explanations!!! Just one sentences."
        return prompt_content

    def initialize(self, prompt_type):
        if(prompt_type == 'cross'):
            prompt_content = ['Please help me create a new algorithm that has a totally different form from the given ones.', \
                              'Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.']
        else:
            prompt_content = ['Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.', \
                              'Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.']
        return prompt_content

    def cross(self,parents):
        #print("Begin: 4")
        ##################################################
        ###  Get algorithm description and code for generating new heuristics as different as possible from parent heuristics
        ##################################################

        # Get prompt to help LLM create algorithms as different as possible from parent heuristics
        prompt_content = self.get_prompt_cross(parents)
        #print("???", prompt_content)
        response = self.interface_llm.get_response(prompt_content)

        # In debug mode, output prompt to help LLM create algorithms as different as possible from parent heuristics
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ cross ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")


        return response

    def variation(self,parents):
        ###########################################################
        ###  Get new heuristic algorithm description and code for modifying the current heuristic to improve performance
        ###########################################################
        prompt_content = self.get_prompt_variation(parents)
        response = self.interface_llm.get_response(prompt_content)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ variation ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")


        return response

#######################################
#######################################
###  How to communicate, how to process algorithms, all prepared, let's start!
#######################################
#######################################
class InterfaceEC_Prompt():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, select,n_p,timeout, problem_type, **kwargs):

        # Set information required by LLM
        self.pop_size = pop_size                   # Define population size

        self.evol = Evolution_Prompt(api_endpoint, api_key, llm_model, debug_mode, problem_type , **kwargs)  # Evolution type, including i1, e1,e2,m1,m2, can be used for algorithm evolution
        self.m = m                                  # Number of parent algorithms for prompt cross operation
        self.debug = debug_mode                     # Whether debug mode is enabled

        # If debug mode is off, no warnings are displayed
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select                        # Parent selection method
        self.n_p = n_p                              # Number of parallel processes
        self.timeout = timeout                      # Timeout definition

    # Write text code to file ./prompt.txt
    def prompt2file(self,prompt):
        with open("./prompt.txt", "w") as file:
        # Write the code to the file
            file.write(prompt)
        return

    # Add a new individual (offspring) to the existing population
    # Precondition: the new individual's objective value should not duplicate any in the population
    # If there is no duplicate objective function value, add it and return True, otherwise return False
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['prompt'] == offspring['prompt']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def extract_first_quoted_string(self, text):
        # Use regular expression to match content within the first double quotes
        match = re.search(r'"(.*?)"', text)
        if match:
            text =  match.group(1)  # Extract the first matched content
        prefix = "Prompt: "
        if text.startswith(prefix):
            return text[len(prefix):].strip()  # Remove prefix and possible leading/trailing spaces
        return text  # If no match, return original string


    # Generate offspring individuals based on the specified evolution operator
    def _get_alg(self,pop,operator):
        #print("Begin: 3")
        # Initialize offspring: Create an offspring dictionary
        offspring = {
            'prompt': None,
            'objective': None,
            'number': None
        }
        off_set = []
        # Get initial prompt
        if operator == "initial_cross":
            # parents = [] # This variable is unused
            prompt_list =  self.evol.initialize("cross")
            for prompt in prompt_list:
                offspring = {
                    'prompt': None,
                    'objective': None,
                    'number': None
                }
                offspring["prompt"] = prompt
                offspring["objective"] = 1e9
                offspring["number"] = []
                off_set.append(offspring)
        elif operator == "initial_variation":
            # parents = [] # This variable is unused
            prompt_list =  self.evol.initialize("variation")
            for prompt in prompt_list:
                offspring = {
                    'prompt': None,
                    'objective': None,
                    'number': None
                }
                offspring["prompt"] = prompt
                offspring["objective"] = 1e9
                offspring["number"] = []
                off_set.append(offspring)
        # Generate new prompt via crossover
        elif operator == "cross":
            parents = self.select.parent_selection(pop,self.m)
            prompt_now = self.evol.cross(parents)
            try:
                prompt_new = self.extract_first_quoted_string(prompt_now)
            except Exception as e:
                print("Prompt cross", e)
                prompt_new = "" # Ensure prompt_new is initialized in case of error
            offspring["prompt"] = prompt_new
            offspring["objective"] = 1e9
            offspring["number"] = []
        # Generate new prompt via mutation
        elif operator == "variation":
            parents = self.select.parent_selection(pop,1)
            #print(parents)
            prompt_now = self.evol.variation(parents)
            try:
                prompt_new = self.extract_first_quoted_string(prompt_now)
            except Exception as e:
                print("Prompt variation", e)
                prompt_new = "" # Ensure prompt_new is initialized in case of error

            offspring["prompt"] = prompt_new
            offspring["objective"] = 1e9
            offspring["number"] = []
        # No such operation!
        else:
            print(f"Prompt operator [{operator}] has not been implemented ! \n")

        # Return selected parent algorithms and generated offspring simultaneously
        return parents, offspring, off_set

    # Generate offspring and evaluate their fitness
    def get_offspring(self, pop, operator):
        try:
            #print("Begin: 2")
            # Call _get_alg method to generate offspring from pop based on operator (i1, m1...) and return parent p and offspring
            #print(operator)
            p, offspring, off_set = self._get_alg(pop, operator)

        # If an exception occurs, set offspring to a dictionary containing all None values, and set p to None
        except Exception as e:
            print("get_offspring", e)
            offspring = {
                'prompt': None,
                'objective': None,
                'number': None
            }
            p = None
            off_set = None

        # Return parent p and generated offspring
        return p, offspring, off_set

    def get_algorithm(self, pop, operator):
        # results: Create an empty list 'results' to store generated offspring
        results = []
        try:
            # Generate self.pop_size offspring. Results are stored in 'results' list, where each element is a (p, off) tuple, with p being the parent and off being the generated offspring.
            if(operator == 'cross' or operator == 'variation'):
                #print("Begin: 1")
                results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size))
            else:
                results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator) for _ in range(1))
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out .")

        time.sleep(2)


        out_p = []   # all parent individuals
        out_off = [] # all offspring individuals

        for p, off, off_set in results:
            out_p.append(p)
            if(operator == 'cross' or operator == 'variation'):
                out_off.append(off)
            else:
                for now_off in off_set:
                    out_off.append(now_off)
            # If in debug mode, print offspring
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off

    def population_generation(self, initial_type):
        # Set to 2, meaning 2 rounds of individuals will be generated
        n_create = 1 # Original was 2, but the loops only run once or twice depending on initial_type for prompt.
        # Create an empty list to store generated initial population individuals
        population = []
        # Loop to generate individuals
        for i in range(n_create):
            _,pop = self.get_algorithm([], initial_type)
            #print(pop)
            for p in pop:
                population.append(p)

        return population


#######################################
#######################################
###  Inner layer! Seems ready? Let's prepare for evolution!
#######################################
#######################################

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode,prompts, backend="auto", **kwargs):

        # set prompt interface
        # getprompts = GetPrompts() # This instance is not needed here if prompts is passed in
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()

        # ["current_node","destination_node","univisited_nodes","distance_matrix"] -> "'current_node','destination_node','univisited_nodes','distance_matrix'"
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        # ["next_node"] -> "'next_node'"
        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # LLM settings
        self.api_endpoint = api_endpoint      # LLM API endpoint for interaction with external service.
        self.api_key = api_key                # API key for authentication and authorization.
        self.model_LLM = model_LLM            # Name or identifier of the language model used.
        self.debug_mode = debug_mode          # Debug mode flag
        self.backend = backend

        # Use the defined InterfaceLLM to set up LLM, then use its get_response(self, prompt_content) function for communication.
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode, self.backend)

    def get_prompt_initial(self):
        #######################################
        ###  Generate initial strategy prompt
        #######################################

        # First, describe the task, then describe what the LLM needs to do:
        # 1. First, describe your new algorithm and its main steps in one sentence. The description must be inside parentheses.
        # 2. Next, implement it in Python as a function named self.prompt_func_name
        # 3. Tell the LLM how many inputs and outputs this function has, and what inputs and outputs (corresponding to processed self.prompt_func_inputs and self.prompt_func_outputs)
        # 4. Describe some properties of the input and output data.
        # 5. Describe some other supplementary properties.
        # 6. Finally, emphasize not to give additional explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content


    def get_prompt_cross(self,indivs, prompt):
        ##################################################
        ###  Generate new heuristics as different as possible from parent heuristics prompt
        ##################################################

        # Combine the algorithm ideas and corresponding codes from individuals into statements like "Algorithm No. 1's idea and code are... Algorithm No. 2's idea and code are..."
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm's thought, objective function value, and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" + str(indivs[i]['objective']) +"\n" +indivs[i]['code']+"\n"
        # 1. Describe the task
        # 2. Tell the LLM how many algorithms we are providing and what they are like (combined with prompt_indiv)
        # 3. Make a request, hoping the LLM creates an algorithm completely different from the previously provided ones
        # 4. Tell the LLM to first describe your new algorithm and main steps in one sentence, with the description enclosed in parentheses.
        # 5. Next, implement it in Python as a function named self.prompt_func_name
        # 6. Tell the LLM how many inputs and outputs this function has, and what inputs and outputs (corresponding to processed self.prompt_func_inputs and self.prompt_func_outputs)
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize not to add extra explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithm's thought, objective function value with their codes as follows: \n"\
+prompt_indiv+ prompt + "\n" +\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content


    def get_prompt_variation(self,indiv1, prompt):
        ##################################################
        ###  Prompt for modifying a heuristic to improve performance
        ##################################################

        # 1. Describe the task
        # 2. Tell the LLM that we are providing one algorithm, describe its idea and code
        # 3. Make a request, hoping the LLM creates a new algorithm with a different form but which can be a modified version of the provided algorithm
        # 4. Tell the LLM to first describe your new algorithm and main steps in one sentence, with the description enclosed in parentheses.
        # 5. Next, implement it in Python as a function named self.prompt_func_name
        # 6. Tell the LLM how many inputs and outputs this function has, and what inputs and outputs (corresponding to processed self.prompt_func_inputs and self.prompt_func_outputs)
        # 7. Describe some properties of the input and output data.
        # 8. Describe some other supplementary properties.
        # 9. Finally, emphasize not to add extra explanations, just output as required.
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n" + \
prompt + "\n" + \
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content



    def _get_alg(self,prompt_content):
        # Get the response for the given prompt_content via the LLM interface.
        #print("QwQ~!")
        response = self.interface_llm.get_response(prompt_content)

        if self.debug_mode:
            print("\n >>> check response for creating algorithm using [ i1 ] : \n", response )
            print(">>> Press 'Enter' to continue")

        # Use regular expression re.findall(r"\{(.*)\}", response, re.DOTALL) to try extracting algorithm description enclosed in curly braces {}.
        # re.DOTALL option allows the regular expression to match newline characters.
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        # If no algorithm description is found within curly braces, use an alternative pattern.
        if len(algorithm) == 0:
            # If the response contains the word 'python', extract the part from the beginning until before the 'python' keyword as the algorithm description.
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            # If it contains 'import', extract the part from the beginning until before 'import' as the algorithm description.
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            # Otherwise, extract the part from the beginning until before 'def' as the algorithm description.
            else:
                algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

        # Try using regular expression re.findall(r"import.*return", response, re.DOTALL) to extract the code part, which starts from 'import' and ends at 'return'.
        code = re.findall(r"import.*return", response, re.DOTALL)
        # If no matching code block is found, try the code block starting from 'def' and ending at 'return'.
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        # If initial extraction of algorithm description or code fails, retry.
        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            # Call get_response method again to get a new response, and repeatedly try to extract algorithm description and code.
            response = self.interface_llm.get_response(prompt_content)
            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            # If the number of retries exceeds 3 (n_retry > 3), exit the loop.
            if n_retry > 3:
                break
            n_retry += 1

        # Assuming the algorithm description and code have been successfully extracted, extract them from the list (i.e., take only the first match).
        algorithm = algorithm[0]
        code = code[0]

        # The extracted code goes up to 'return', we complete the rest. Connect the extracted code with output variables (stored in self.prompt_func_outputs) to form a complete code string.
        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs)


        return [code_all, algorithm]


    def initial(self):
        ##################################################
        ###  Get algorithm description and code for creating the initial population
        ##################################################

        # Get the prompt to help LLM create the initial population
        prompt_content = self.get_prompt_initial()

        # In debug mode, output the prompt to help LLM create the initial population
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")

        # Call _get_alg, input the prompt to LLM, and split the returned text into code and algorithm description
        [code_all, algorithm] = self._get_alg(prompt_content)

        # In debug mode, split the returned text into code and algorithm description
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")


        return [code_all, algorithm]

    def cross(self, parents, prompt):
        ##################################################
        ###  Get algorithm description and code for generating new heuristics as different as possible from parent heuristics
        ##################################################

        # Get prompt to help LLM create algorithms as different as possible from parent heuristics
        prompt_content = self.get_prompt_cross(parents, prompt)

        # In debug mode, output prompt to help LLM create algorithms as different as possible from parent heuristics
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")

        # Call _get_alg, input the prompt to LLM, and split the returned text into code and algorithm description
        [code_all, algorithm] = self._get_alg(prompt_content)

        # In debug mode, split the returned text into code and algorithm description
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")


        return [code_all, algorithm]

    def variation(self,parents, prompt):
        ###########################################################
        ###  Get new heuristic algorithm description and code for modifying the current heuristic to improve performance
        ###########################################################
        prompt_content = self.get_prompt_variation(parents, prompt)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")


        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")


        return [code_all, algorithm]

############################################################################################################
###  Adds an import statement to the given Python code.
###  This function inserts an 'import package_name as as_name' statement into the code, if the package is not already imported.
############################################################################################################
def add_import_package_statement(program: str, package_name: str, as_name=None, *, check_imported=True) -> str:
    """Add 'import package_name as as_name' in the program code.
    """
    # Use ast.parse() method to parse the input Python code string into an Abstract Syntax Tree (AST).
    tree = ast.parse(program)

    # If check_imported is True, traverse each node in the AST to check if there is already an import statement for the specified package.
    if check_imported:
        # check if 'import package_name' code exists
        package_imported = False
        for node in tree.body:
            # The first part checks if the node is an import statement.
            # The second part checks if the package name in the import statement is the same as package_name.
            if isinstance(node, ast.Import) and any(alias.name == package_name for alias in node.names):
                package_imported = True
                break
        # If the package is already imported, the package_imported flag is set to True, and the unmodified code is returned directly.
        if package_imported:
            return ast.unparse(tree)

    # Create a new import node. Use ast.Import to create a new import statement node with package_name as the name and as_name as the alias (if any).
    import_node = ast.Import(names=[ast.alias(name=package_name, asname=as_name)])
    # Insert the new import node at the very top of the AST.
    tree.body.insert(0, import_node)
    # Use ast.unparse(tree) to convert the modified AST back into a Python code string and return it.
    program = ast.unparse(tree)
    return program


############################################################################################################
###  Adds NumPy's @numba.jit(nopython=True) decorator to the given Python code.
###  The decorator is added to a specified function to improve its execution efficiency.
###  Numba is a JIT (Just-In-Time) compiler for accelerating numerical computations.
############################################################################################################
def _add_numba_decorator(
        program: str,
        function_name: str
) -> str:
    # Use ast.parse() method to parse the input Python code string into an Abstract Syntax Tree (AST).
    tree = ast.parse(program)

    # Traverse each node in the AST to check if an 'import numba' statement already exists.
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    # If numba is not yet imported, create an 'import numba' node and insert it at the very top of the AST.
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)


    for node in ast.walk(tree):
        # Use ast.walk(tree) to traverse all nodes in the AST, finding function definitions that match function_name.
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Create an @numba.jit(nopython=True) decorator node.
            # ast.Call creates a call node, ast.Attribute represents attribute access (i.e., numba.jit), ast.keyword creates a keyword node with named arguments.
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit',
                    ctx=ast.Load()
                ),
                args=[],  # args do not have argument name
                keywords=[ast.keyword(arg='nopython', value=ast.NameConstant(value=True))]
                # keywords have argument name
            )
            # Add it to the target function's decorator_list attribute.
            node.decorator_list.append(decorator)

    # Use ast.unparse(tree) to convert the modified AST back into a Python code string and return it.
    modified_program = ast.unparse(tree)
    return modified_program


def add_numba_decorator(
        program: str,
        function_name: str | Sequence[str],
) -> str:
    # If function_name is a string, it means only one function needs a decorator. In this case, call the helper function _add_numba_decorator and return its result.
    if isinstance(function_name, str):
        return _add_numba_decorator(program, function_name)
    # If function_name is a sequence (e.g., list or tuple), iterate through each function name. For each function name, call _add_numba_decorator and update the program.
    for f_name in function_name:
        program = _add_numba_decorator(program, f_name)
    return program


############################################################################################################
## Add fixed random seed, at the beginning and in functions.
############################################################################################################
# Inserts a statement to set the random seed `np.random.seed(...)` into the specified Python code.
# If the `numpy` module (i.e., `import numpy as np` statement) is not yet imported in the code, this function first adds that import statement.
def add_np_random_seed_below_numpy_import(program: str, seed: int = 2024) -> str:
    # This line calls the `add_import_package_statement` function to ensure `import numpy as np` is present in the program.
    program = add_import_package_statement(program, 'numpy', 'np')
    # Use Python's `ast` (Abstract Syntax Tree) module to parse the code into a syntax tree.
    tree = ast.parse(program)

    # find 'import numpy as np'
    found_numpy_import = False

    # find 'import numpy as np' statement
    for node in tree.body:
        # Loop through the nodes of the syntax tree, looking for the `import numpy as np` statement.
        if isinstance(node, ast.Import) and any(alias.name == 'numpy' and alias.asname == 'np' for alias in node.names):
            found_numpy_import = True
            # Insert `np.random.seed(seed)` statement after the found `import numpy as np` statement. This is achieved by creating a new AST node representing a call to the `np.random.seed` function.
            node_idx = tree.body.index(node)
            seed_node = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id='np', ctx=ast.Load()),
                            attr='random',
                            ctx=ast.Load()
                        ),
                        attr='seed',
                        ctx=ast.Load()
                    ),
                    args=[ast.Num(n=seed)],
                    keywords=[]
                )
            )
            tree.body.insert(node_idx + 1, seed_node)
    # If no `import numpy as np` statement is found in the code, raise a `ValueError` exception. This step ensures that numpy is imported before `np.random.seed(seed)` is inserted.
    if not found_numpy_import:
        raise ValueError("No 'import numpy as np' found in the code.")
    # Use `ast.unparse` method to convert the modified syntax tree back into a Python code string, and return the string.
    modified_code = ast.unparse(tree)
    return modified_code

# Adds an `np.random.seed(seed)` statement within the specified Python function to set the random number generator's seed.
# This operation is typically used to ensure reproducibility of random number generation across different runs. Below is a detailed explanation of this code.
def add_numpy_random_seed_to_func(program: str, func_name: str, seed: int = 2024) -> str:
    # This line parses the input code string into an Abstract Syntax Tree (AST).
    tree = ast.parse(program)

    # Inserts the new `np.random.seed(seed)` statement at the beginning of the target function's body.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            node.body = [ast.parse(f'np.random.seed({seed})').body[0]] + node.body

    # Converts the entire syntax tree into a new code string containing the seed setting.
    modified_code = ast.unparse(tree)
    return modified_code

############################################################################################################
###  Replaces the division operator (/) in Python code with a custom protected division function `_protected_div`, and optionally uses Numba for acceleration.
############################################################################################################
# First, define a `_CustomDivisionTransformer` class, which inherits from `ast.NodeTransformer`.
# Its purpose is to traverse the Abstract Syntax Tree (AST), find all division operations (/), and replace them with calls to a custom division function.
class _CustomDivisionTransformer(ast.NodeTransformer):
    # It accepts a parameter `custom_divide_func_name`, which represents the name of the custom function to be used to replace the division operator. Here, the name is `_protected_div`.
    def __init__(self, custom_divide_func_name: str):
        super().__init__()
        self._custom_div_func = custom_divide_func_name

    # Used to visit all binary operator nodes. If a division operator (/) is detected, it is replaced with the custom function.
    def visit_BinOp(self, node):
        self.generic_visit(node)  # recur visit child nodes
        if isinstance(node.op, ast.Div):
            # self-defined node
            custom_divide_call = ast.Call(
                func=ast.Name(id=self._custom_div_func, ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            return custom_divide_call
        return node

# Its purpose is to replace all division operators in the input code string with a custom protected division function named `_protected_div`, which avoids division by zero.
def replace_div_with_protected_div(code_str: str, delta=1e-5, numba_accelerate=False) -> Tuple[str, str]:
    # Define the protected division function `_protected_div`.
    protected_div_str = f'''
def _protected_div(x, y, delta={delta}):
    return x / (y + delta)
    '''
    # Parse the input code string into an AST tree.
    tree = ast.parse(code_str)

    # Create a `_CustomDivisionTransformer` instance and traverse the AST to find and replace division operations.
    transformer = _CustomDivisionTransformer('_protected_div')
    modified_tree = transformer.visit(tree)

    # Convert the modified AST tree back to a code string, returning both the modified code and the protected division function definition.
    modified_code = ast.unparse(modified_tree)
    modified_code = '\n'.join([modified_code, '', '', protected_div_str])

    # If `numba_accelerate` is true, add the `@numba.jit()` decorator to the `_protected_div` function for acceleration.
    if numba_accelerate:
        modified_code = add_numba_decorator(modified_code, '_protected_div')
    # Return the modified code string and the name of the custom division function.
    return modified_code, '_protected_div'

#######################################
#######################################
###  How to communicate, how to process algorithms, all prepared, let's start!
#######################################
#######################################
class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # Set information required by LLM
        self.pop_size = pop_size                    # Define population size
        self.interface_eval = interface_prob        # PROBLEMCONST() type, can call the evaluate function to assess algorithm code
        prompts = interface_prob.prompts            # Problem description, input/output information prompts, which can be used to generate subsequent prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode,prompts, **kwargs)  # Evolution type, including i1, e1,e2,m1,m2, can be used for algorithm evolution
        self.m = m                                  # Number of parent algorithms for 'e1' and 'e2' operations
        self.debug = debug_mode                     # Whether debug mode is enabled

        # If debug mode is off, no warnings are displayed
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select                        # Parent selection method
        self.n_p = n_p                              # Number of parallel processes

        self.timeout = timeout                      # Timeout definition
        #self.timeout = 400 # This line was commented out in the original
        self.use_numba = use_numba                  # Whether to use Numba library for acceleration

    def _typed_seed_program(self, description, body_lines):
        lines = [
            "import numpy as np",
            "",
            f'"""{description}"""',
            "def select_neighborhood(n, m, k, site, value, constraint, initial_solution, current_solution, objective_coefficient, global_features, variable_features, constraint_features, neighborhood_context, operator_performance, decision_context):",
        ]
        for line in body_lines:
            lines.append("    " + line)
        return "\n".join(lines)

    def get_typed_operator_seed_algorithms(self):
        seeds = []
        seeds.append({
            "algorithm": "Typed FRAC-LNS baseline that prioritizes highly fractional variables when the LP gap is large and falls back to a slightly wider release ratio in repair mode.",
            "code": self._typed_seed_program(
                "Typed FRAC-LNS baseline that releases highly fractional variables and adapts free_ratio to LP gap and repair pressure.",
                [
                    "base_ratio = 0.12 if neighborhood_context.get('neighborhood_mode', 0) == 0 else 0.18",
                    "free_ratio = min(0.3, max(base_ratio, 0.08 + 0.4 * global_features.get('lp_gap', 0.0)))",
                    "return TypedNeighborhoodOperatorLibrary.make_operator(",
                    "    'FRAC-LNS',",
                    "    free_ratio=free_ratio,",
                    "    time_budget=120,",
                    ")",
                ],
            ),
        })
        seeds.append({
            "algorithm": "Typed TIGHT-LNS baseline that focuses on tight or nearly violated constraints and slightly enlarges the neighborhood when repair is needed.",
            "code": self._typed_seed_program(
                "Typed TIGHT-LNS baseline that releases variables around the tightest constraints.",
                [
                    "free_ratio = 0.14 if neighborhood_context.get('neighborhood_mode', 0) == 0 else 0.22",
                    "return TypedNeighborhoodOperatorLibrary.make_operator(",
                    "    'TIGHT-LNS',",
                    "    free_ratio=free_ratio,",
                    "    time_budget=120,",
                    ")",
                ],
            ),
        })
        seeds.append({
            "algorithm": "Typed OBJ-LNS baseline that targets variables with large objective leverage and large LP-incumbent discrepancy.",
            "code": self._typed_seed_program(
                "Typed OBJ-LNS baseline that focuses on high-impact objective variables.",
                [
                    "free_ratio = 0.10 if global_features.get('no_improve_rounds', 0.0) < 2 else 0.16",
                    "return TypedNeighborhoodOperatorLibrary.make_operator(",
                    "    'OBJ-LNS',",
                    "    free_ratio=free_ratio,",
                    "    time_budget=90,",
                    ")",
                ],
            ),
        })
        seeds.append({
            "algorithm": "Typed HYBRID-LNS baseline that combines tight-constraint pressure, LP fractionality, and diversity to balance exploitation and exploration.",
            "code": self._typed_seed_program(
                "Typed HYBRID-LNS baseline that mixes TIGHT-LNS, FRAC-LNS, and DIVERSITY-LNS.",
                [
                    "repair_mode = neighborhood_context.get('neighborhood_mode', 0) == 1",
                    "diversity_weight = 0.3 if global_features.get('no_improve_rounds', 0.0) >= 2 else 0.15",
                    "tight_weight = 0.45 if not repair_mode else 0.35",
                    "frac_weight = 0.35 if global_features.get('lp_gap', 0.0) >= 0.08 else 0.25",
                    "components = [",
                    "    {'name': 'TIGHT-LNS', 'weight': tight_weight},",
                    "    {'name': 'FRAC-LNS', 'weight': frac_weight},",
                    "    {'name': 'DIVERSITY-LNS', 'weight': diversity_weight},",
                    "]",
                    "return TypedNeighborhoodOperatorLibrary.make_operator(",
                    "    'HYBRID-LNS',",
                    "    components=components,",
                    "    free_ratio=0.15 if not repair_mode else 0.22,",
                    "    time_budget=120,",
                    ")",
                ],
            ),
        })
        seeds.append({
            "algorithm": "Typed GRAPH-BLOCK/HISTORY hybrid baseline that exploits graph-local neighborhoods after some search history accumulates and otherwise starts from sparse local blocks.",
            "code": self._typed_seed_program(
                "Typed baseline that uses GRAPH-BLOCK-LNS early and HISTORY-LNS later.",
                [
                    "if global_features.get('time_elapsed', 0.0) <= 1.0 or global_features.get('no_improve_rounds', 0.0) < 1:",
                    "    return TypedNeighborhoodOperatorLibrary.make_operator(",
                    "        'GRAPH-BLOCK-LNS',",
                    "        free_ratio=0.12 if global_features.get('density', 1.0) < 0.15 else 0.1,",
                    "        time_budget=90,",
                    "    )",
                    "return TypedNeighborhoodOperatorLibrary.make_operator(",
                    "    'HISTORY-LNS',",
                    "    free_ratio=0.14 if neighborhood_context.get('neighborhood_mode', 0) == 0 else 0.2,",
                    "    time_budget=90,",
                    ")",
                ],
            ),
        })
        return seeds

    # Write text code to file ./ael_alg.py
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return

    # Add a new individual (offspring) to the existing population
    # Precondition: the new individual's objective value should not duplicate any in the population
    # If there is no duplicate objective function value, add it and return True, otherwise return False
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    # Checks if the given code snippet (code) already exists in any individual within the population.
    # By checking for duplicate code snippets, redundant additions of identical individuals to the population can be avoided.
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    # Generates offspring individuals based on the specified evolution operator.
    def _get_alg(self,pop,operator, prompt):
        # Initialize offspring: Create an offspring dictionary
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        # Get initial algorithm
        if operator == "initial":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.initial()
        # Generate algorithms dissimilar to parents
        elif operator == "cross":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.cross(parents, prompt)
        # Generate new algorithms by improving current ones
        elif operator == "variation":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.variation(parents[0], prompt)
        # No such operation!
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        # Return selected parent algorithms and generated offspring simultaneously
        return parents, offspring

    # Generates offspring and evaluate their fitness
    def get_offspring(self, pop, operator, prompt):

        try:
            # Call _get_alg method to generate offspring from pop based on operator (i1, m1...) and return parent p and offspring
            p, offspring = self._get_alg(pop, operator, prompt)

            # Whether to use Numba
            if self.use_numba:
                # Use regular expression r"def\s+(\w+)\s*\(.*\):" to match function definitions
                pattern = r"def\s+(\w+)\s*\(.*\):"
                # Extract function name from offspring['code']
                match = re.search(pattern, offspring['code'])
                function_name = match.group(1)
                # Call add_numba_decorator method to add Numba decorator to the function
                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            # Handle duplicate code
            n_retry= 1
            while self.check_duplicate(pop, offspring['code']):
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")

                # If the generated code duplicates existing code in the population, regenerate offspring
                p, offspring = self._get_alg(pop, operator, prompt) # Missing prompt here
                if p is None and offspring['code'] is None: # Break if generation itself failed
                    break

                # Whether to use Numba
                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']

                # Attempt at most once
                if n_retry > 1: # Original code was n_retry > 1, meaning it would only retry once (attempt 2). If it's `>1`, it means 2nd attempt onwards.
                    break

            # If generation failed or retries exhausted for duplicate code, ensure offspring is marked as invalid
            if offspring['code'] is None:
                raise ValueError("Failed to generate valid offspring code after retries.")

            # Create thread pool: Use ThreadPoolExecutor to execute evaluation tasks
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit self.interface_eval.evaluate method for evaluation, passing the generated code
                future = executor.submit(self.interface_eval.evaluate, code)
                # Get evaluation result fitness, round it to 5 decimal places, and store it in offspring['objective']
                fitness = future.result(timeout=self.timeout)
                offspring['objective'] = np.round(fitness, 5)
                # End task to release resources
                future.cancel()

        # If an exception occurs, set offspring to a dictionary containing all None values, and set p to None
        except Exception as e:
            print(f"Error in get_offspring: {e}")
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None

        # Return parent p and generated offspring
        return p, offspring

    def get_algorithm(self, pop, operator, prompt):
        # results: Create an empty list 'results' to store generated offspring
        results = []
        try:
            # Generate self.pop_size offspring. Results are stored in 'results' list, where each element is a (p, off) tuple, with p being the parent and off being the generated offspring.
            results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator, prompt) for _ in range(self.pop_size))
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out .")

        time.sleep(2)


        out_p = []   # all parent individuals
        out_off = [] # all offspring individuals

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            # If in debug mode, print offspring
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off

    def population_generation(self):
        population = []
        typed_seeds = self.get_typed_operator_seed_algorithms()
        if typed_seeds:
            seed_count = min(len(typed_seeds), self.pop_size)
            population.extend(
                self.population_generation_seed(
                    typed_seeds[:seed_count],
                    min(max(1, self.n_p), seed_count),
                )
            )

        n_missing = max(self.pop_size - len(population), 0)
        if n_missing == 0:
            return population

        # Fill the remaining slots with LLM-generated initial heuristics.
        while len(population) < self.pop_size:
            _, pop = self.get_algorithm([], 'initial', [])
            for p in pop:
                if p["code"] is None or self.check_duplicate(population, p["code"]):
                    continue
                population.append(p)
                if len(population) >= self.pop_size:
                    break
        return population

    # Generates a population based on seeds (recorded algorithms), where each individual's fitness is obtained through parallel evaluation.
    def population_generation_seed(self,seeds,n_p):
        # Create an empty list to store generated population individuals.
        population = []
        # Evaluate the code of each seed using self.interface_eval.evaluate method and calculate its fitness.
        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)
        # Iterate through each seed and its corresponding fitness.
        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print(f"Initialization finished! Get {len(seeds)} seed algorithms")

        return population


class EOH:

    # Initialization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem      # Define the problem
        self.select = select     # Define parent selection method
        self.manage = manage     # Define population management method

        # LLM settings
        self.api_endpoint = paras.llm_api_endpoint  # Define API endpoint URL for communication with the language model
        self.api_key = paras.llm_api_key            # API private key
        self.llm_model = paras.llm_model            # Define the large language model to use
        self.llm_backend = paras.llm_backend        # remote or local openai-compatible backend

        # prompt
        self.pop_size_cross = 2
        self.pop_size_variation = 2
        self.problem_type = "minimization"

        # Experimental settings
        self.pop_size = paras.ec_pop_size  # Population size
        self.n_pop = paras.ec_n_pop        # Number of populations/generations to run

        self.operators = paras.ec_operators   # Define number of operators, default is four: e1, e2, m1, m2

        self.operator_weights = paras.ec_operator_weights    # Define operator weights [0, 1], higher weight means higher probability of using the operator
        if paras.ec_m > self.pop_size or paras.ec_m == 1:    # Number of parents required for e1 and e2 operations, at least two but not exceeding population size
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m                                  # Set number of parents required for e1 and e2 operations

        self.debug_mode = paras.exp_debug_mode               # Whether debug mode is enabled
        # self.ndelay = 1  # default # This variable is not used in the original code.

        self.output_path = paras.exp_output_path             # Population results saving path

        self.exp_n_proc = paras.exp_n_proc                   # Number of processes set

        self.timeout = paras.eva_timeout                     # Timeout definition

        self.prompt_timeout = paras.prompt_eva_timeout

        self.use_numba = paras.eva_numba_decorator           # Whether to use Numba library for acceleration

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # Add newly generated offspring to the population; if debug mode is enabled, compare them with existing individuals for redundancy.
    def add2pop(self, population, offspring):
        for off in offspring:
            is_duplicated = False
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
                    is_duplicated = True
                    break
            if not is_duplicated:
                population.append(off)

    def add2pop_prompt(self, population, offspring):
        for off in offspring:
            is_duplicated = False
            for ind in population:
                if ind['prompt'] == off['prompt']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
                    is_duplicated = True
                    break
            if not is_duplicated:
                population.append(off)


    # Run EOH
    def run(self):

        print("- Evolution Start -")
        # Record start time
        time_start = time.time()

        # Set problem evaluation window
        interface_prob = self.prob

        # Set prompt evolution
        interface_promt_cross = InterfaceEC_Prompt(self.pop_size_cross, self.m, self.api_endpoint, self.api_key, self.llm_model, self.debug_mode, self.select, self.exp_n_proc, self.prompt_timeout, self.problem_type, backend=self.llm_backend)
        interface_promt_variation = InterfaceEC_Prompt(self.pop_size_variation, self.m, self.api_endpoint, self.api_key, self.llm_model, self.debug_mode, self.select, self.exp_n_proc, self.prompt_timeout, self.problem_type, backend=self.llm_backend)
        # Set evolution mode, including initialization, evolution, and management
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba, backend=self.llm_backend
                                   )

        # Initialize the population
        cross_operators = []
        variation_operators = []
        print("creating initial prompt:")
        cross_operators = interface_promt_cross.population_generation("initial_cross")
        # cross_operators = self.manage.population_management(cross_operators, self.pop_size_cross) # This line was commented out in the original and is duplicated below
        variation_operators = interface_promt_variation.population_generation("initial_variation")
        # variation_operators = self.manage.population_management(variation_operators, self.pop_size_variation) # This line was commented out in the original and is duplicated below
        print(f"Prompt initial: ")

        for prompt in cross_operators:
            print("Cross Prompt: ", prompt['prompt'])
        for prompt in variation_operators:
            print("Variation Prompt: ", prompt['prompt'])
        print("initial population has been created!")


        print("=======================================")
        population = []
        print("creating initial population:")
        population = interface_ec.population_generation()
        population = self.manage.population_management(population, self.pop_size)

        print(f"Pop initial: ")
        for off in population:
            print(" Obj: ", off['objective'], end="|")
        print()
        print("initial population has been created!")
        # Save the generated population to a file
        filename = self.output_path + "/results/pops/population_generation_0.json"
        with open(filename, 'w') as f:
            json.dump(population, f, indent=5)
        n_start = 0

        print("=======================================")

        # n_op: number of evolution operations
        n_op = len(self.operators)
        worst = []
        delay_turn = 3
        change_flag = 0
        last = -1
        max_k = 4
        # n_pop: number of populations/generations to run
        for pop_idx in range(n_start, self.n_pop):
            #print(f" [{na + 1} / {self.pop_size}] ", end="|") # `na` is not defined

            if(change_flag):
                change_flag -= 1
                if(change_flag == 0):
                    cross_operators = self.manage.population_management(cross_operators, self.pop_size_cross)
                    for prompt in cross_operators:
                        print("Cross Prompt: ", prompt['prompt'])

                    variation_operators = self.manage.population_management(variation_operators, self.pop_size_variation)
                    for prompt in variation_operators:
                        print("Variation Prompt: ", prompt['prompt'])

            if(len(worst) >= delay_turn and worst[-1] == worst[-delay_turn] and pop_idx - last > delay_turn):
                parents_cross, offsprings_cross = interface_promt_cross.get_algorithm(cross_operators, 'cross')
                self.add2pop_prompt(cross_operators, offsprings_cross)

                parents_variation_cross, offsprings_variation_cross = interface_promt_cross.get_algorithm(cross_operators, 'variation')
                self.add2pop_prompt(cross_operators, offsprings_variation_cross)

                for prompt in cross_operators:
                    print("Cross Prompt: ", prompt['prompt'])
                    prompt["objective"] = 1e9
                    prompt["number"] = []

                parents_variation, offsprings_variation = interface_promt_variation.get_algorithm(variation_operators, 'cross')
                self.add2pop_prompt(variation_operators, offsprings_variation)

                parents_cross_variation, offsprings_cross_variation = interface_promt_variation.get_algorithm(variation_operators, 'variation')
                self.add2pop_prompt(variation_operators, offsprings_cross_variation)

                for prompt in variation_operators:
                    print("Variation Prompt: ", prompt['prompt'])
                    prompt["objective"] = 1e9
                    prompt["number"] = []

                change_flag = 2
                last = pop_idx

            # Let's look at the crossover operation first
            for i in range(len(cross_operators)):
                current_prompt = cross_operators[i]["prompt"]
                print(f" OP: cross, [{i + 1} / {len(cross_operators)}] ", end="|")
                parents, offsprings = interface_ec.get_algorithm(population, "cross", current_prompt)
                # Add newly generated offspring to the population; if debug mode is enabled, compare them with existing individuals for redundancy.
                self.add2pop(population, offsprings)
                for off in offsprings:
                    print(" Obj: ", off['objective'], end="|")
                    if(off['objective'] is None):
                        continue

                    if len(cross_operators[i]["number"]) < max_k:
                        heapq.heappush(cross_operators[i]["number"], -off['objective'])
                    else:
                        # If the heap is full and the current element is smaller than the heap top, replace the heap top.
                        if off['objective'] < -cross_operators[i]["number"][0]:
                            heapq.heapreplace(cross_operators[i]["number"], -off['objective'])  # Replace heap top element

                    cross_operators[i]["objective"] = -sum(cross_operators[i]["number"]) / len(cross_operators[i]["number"])
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop_idx + 1) + "_" + str( # na is not defined
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)

                # With new offspring added, population size might exceed; manage the population to keep size at most pop_size
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f"Cross {i + 1}, objective: {cross_operators[i]['objective']}", end = "|")
                print()

            # Now let's look at the mutation operation
            for i in range(len(variation_operators)): # Loop over variation_operators, not cross_operators
                current_prompt = variation_operators[i]["prompt"]
                print(f" OP: variation, [{i + 1} / {len(variation_operators)}] ", end="|")
                parents, offsprings = interface_ec.get_algorithm(population, "variation", current_prompt)
                # Add newly generated offspring to the population; if debug mode is enabled, compare them with existing individuals for redundancy.
                self.add2pop(population, offsprings)
                for off in offsprings:
                    print(" Obj: ", off['objective'], end="|")
                    if(off['objective'] is None):
                        continue
                    if len(variation_operators[i]["number"]) < max_k:
                        heapq.heappush(variation_operators[i]["number"], -off['objective'])
                    else:
                        # If the heap is full and the current element is smaller than the heap top, replace the heap top.
                        if off['objective'] < -variation_operators[i]["number"][0]:
                            heapq.heapreplace(variation_operators[i]["number"], -off['objective'])  # Replace heap top element

                    variation_operators[i]["objective"] = -sum(variation_operators[i]["number"]) / len(variation_operators[i]["number"])
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop_idx + 1) + "_" + str( # na is not defined
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)

                # With new offspring added, population size might exceed; manage the population to keep size at most pop_size
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f"variation {i + 1}, objective: {variation_operators[i]['objective']}", end = "|")
                print()

            '''
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                op_w = self.operator_weights[i]
                # If the random number is less than the weight (weight range is 0 to 1), then run
                # In other words, for each operation, its operator_weights is the probability of running this operation
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op, current_prompt) # current_prompt is not defined in this scope
                # Add newly generated offspring to the population; if debug mode is enabled, compare them with existing individuals for redundancy.
                self.add2pop(population, offsprings)
                for off in offsprings:
                    print(" Obj: ", off['objective'], end="|")
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop_idx + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)

                # With new offspring added, population size might exceed; manage the population to keep size at most pop_size
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()
            '''

            # Save the population to a file, each generation has its own file
            filename = self.output_path + "/results/pops/population_generation_" + str(pop_idx + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best individual of the population to a file, each generation has its own file
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop_idx + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            # Output time in minutes
            print(f"--- {pop_idx + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            # Output the objective function values of the remaining population after management
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            worst.append(population[-1]['objective'])
            print()


class Methods():
    # Set parent selection and population management methods, which is quite ingenious to map strings to function methods
    def __init__(self,paras,problem) -> None:
        self.paras = paras
        self.problem = problem
        if paras.selection == "prob_rank":
            self.select = prob_rank
        elif paras.selection == "equal":
            self.select = equal
        elif paras.selection == 'roulette_wheel':
            self.select = roulette_wheel
        elif paras.selection == 'tournament':
            self.select = tournament
        else:
            print("selection method "+paras.selection+" has not been implemented !")
            exit()

        if paras.management == "pop_greedy":
            self.manage = pop_greedy
        elif paras.management == 'ls_greedy':
            self.manage = ls_greedy
        elif paras.management == 'ls_sa':
            self.manage = ls_sa
        else:
            print("management method "+paras.management+" has not been implemented !")
            exit()


    def get_method(self):
        # Must run EoH, right?
        if self.paras.method == "eoh":
            return EOH(self.paras,self.problem,self.select,self.manage)
        else:
            print("method "+self.paras.method+" has not been implemented!")
            exit()

class EVOL:
    # Initialization
    def __init__(self, paras, prob=None, **kwargs):

        print("----------------------------------------- ")
        print("---              Start EoH            ---")
        print("-----------------------------------------")
        # Create folder #
        create_folders(paras.exp_output_path)
        print("- output folder created -")

        self.paras = paras

        print("-  parameters loaded -")

        self.prob = prob

        # Set a random seed
        random.seed(2024)


    # Run methods
    def run(self):

        problemGenerator = Probs(self.paras)

        problem = problemGenerator.get_problem()

        methodGenerator = Methods(self.paras,problem)

        method = methodGenerator.get_method()

        method.run()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")

def build_default_paras(problem_config=None):
    configure_problem(problem_config)
    default_endpoint = os.environ.get("LLM_SAS_LLM_ENDPOINT", "http://127.0.0.1:8000")
    default_key = os.environ.get("LLM_SAS_LLM_API_KEY", "EMPTY")
    default_model = os.environ.get("LLM_SAS_LLM_MODEL", "your_local_model_name")
    default_backend = os.environ.get("LLM_SAS_LLM_BACKEND", "auto")
    paras = Paras()
    paras.set_paras(method = "eoh",    # ['ael','eoh']
                    problem = "milp_construct", #['milp_construct','bp_online']
                    llm_api_endpoint = default_endpoint, # remote host or local OpenAI-compatible base URL
                    llm_api_key = default_key,   # use a real key for remote APIs; local servers often ignore this
                    llm_model = default_model,
                    llm_backend = default_backend,
                    ec_pop_size = 4, # number of samples in each population
                    ec_n_pop = 20,  # number of populations
                    exp_n_proc = 4,  # multi-core parallel
                    exp_debug_mode = False,
                    exp_output_path = ACTIVE_PROBLEM_CONFIG["exp_output_path"])
    return paras


def main(problem_config=None):
    paras = build_default_paras(problem_config)
    evolution = EVOL(paras)
    evolution.run()


if __name__ == "__main__":
    main()
