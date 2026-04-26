from milp_problem_eoh_common import main


PROBLEM_CONFIG = {
    "problem_code": "SC",
    "problem_label": "Set Cover",
    "problem_prompt_description": "This corresponds to the Set Cover MILP problem, where all decision variables are binary (0-1 variables), and all constraints are in the form of LHS >= RHS.",
    "instance_path": "./SC_easy_instance/LP",
    "exp_output_path": "./ACP_SC/",
}


if __name__ == "__main__":
    main(PROBLEM_CONFIG)
