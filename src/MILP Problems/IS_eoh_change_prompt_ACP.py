from milp_problem_eoh_common import main


PROBLEM_CONFIG = {
    "problem_code": "IS",
    "problem_label": "Independent Set",
    "problem_prompt_description": "This corresponds to the Independent Set MILP problem, where all decision variables are binary (0-1 variables), and all constraints are in the form of LHS <= RHS.",
    "instance_path": "./IS_easy_instance/LP",
    "exp_output_path": "./IS_ACP/",
}


if __name__ == "__main__":
    main(PROBLEM_CONFIG)
