from milp_problem_eoh_common import main


PROBLEM_CONFIG = {
    "problem_code": "MVC",
    "problem_label": "Minimum Vertex Cover",
    "problem_prompt_description": "This corresponds to the Minimum Vertex Cover MILP problem, where all decision variables are binary (0-1 variables).",
    "instance_path": "./MVC_easy_instance/LP",
    "exp_output_path": "./ACP_MVC/",
}


if __name__ == "__main__":
    main(PROBLEM_CONFIG)
