from milp_problem_eoh_common import main


PROBLEM_CONFIG = {
    "problem_code": "MIKS",
    "problem_label": "Maximum Independent K-Set",
    "problem_prompt_description": "This corresponds to the Maximum Independent K-Set MILP problem, where all constraints are in the form of LHS <= RHS or other standard MILP forms.",
    "instance_path": "./MIKS_easy_instance/LP",
    "exp_output_path": "./MIKS2_ACP/",
}


if __name__ == "__main__":
    main(PROBLEM_CONFIG)
