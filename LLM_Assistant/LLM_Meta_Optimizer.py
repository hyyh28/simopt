from langchain_openai import ChatOpenAI
from LLM import url, key
from simopt.directory import problem_directory, model_introduction_directory, problem_unabbreviated_directory, solver_introduction_directory, solver_directory, solver_unabbreviated_directory, model_directory, model_unabbreviated_directory, model_problem_unabbreviated_directory, model_problem_class_directory
from simopt.experiment_base import ProblemSolver, ProblemsSolvers, post_normalize, find_missing_experiments, make_full_metaexperiment, plot_progress_curves, plot_solvability_cdfs, plot_area_scatterplots, plot_solvability_profiles, plot_terminal_progress, plot_terminal_scatterplots
from langchain_core.prompts import PromptTemplate

user_prompt_of_introduction = PromptTemplate.from_template("You choose {problem} based on {model} model, which is {model_introduction}")
llm_prompt_of_introduction = PromptTemplate.from_template("The user choose {problem} based on {model} model, which is {model_introduction}")
problem_setting_prompt = PromptTemplate.from_template("The parameters of {problem} is:\n")
solver_setting_prompt = PromptTemplate.from_template("The parameters of {solver} is:\n")
prompt_of_methods_introduction = PromptTemplate.from_template("Please choose one method to solve the problem: {problem_choice}\n Followed with their introduction:\n")

def get_problem_introduction(problem_name="Min Deterministic Function + Noise (SUCG)"):
    model_of_problem = model_problem_unabbreviated_directory[problem_name]
    introduction_of_model = model_introduction_directory[model_of_problem]
    llm_problem_introduction = llm_prompt_of_introduction.format(problem=problem_name, model=model_of_problem,
                                                                 model_introduction=introduction_of_model)
    user_problem_introduction = llm_prompt_of_introduction.format(problem=problem_name, model=model_of_problem,
                                                                 model_introduction=introduction_of_model)
    return llm_problem_introduction, user_problem_introduction

def get_problem_setting(problem_name="Min Deterministic Function + Noise (SUCG)"):
    problem_class = problem_unabbreviated_directory[problem_name]
    problem = problem_class()
    problem_setting_description = problem_setting_prompt.format(problem=problem_name)
    for factor in problem.factors.keys():
        problem_setting_description += str(factor) + ":" + str(problem.factors[factor]) + "\n"
    return problem_setting_description

def get_methods_list_and_introduction():
    solvers = solver_introduction_directory.keys()
    solvers = str(list(solvers))[1:-1]
    llm_method_introduction = str(prompt_of_methods_introduction.format(problem_choice=solvers))
    for solver in solver_introduction_directory.keys():
        solver_introduction = solver_introduction_directory[solver] + "\n"
        llm_method_introduction += solver_introduction
    return llm_method_introduction


def get_methods_parameters(method="ASTRO-DF (SBCN)"):
    solver_class = solver_unabbreviated_directory[method]
    solver = solver_class()
    specifications = solver.specifications
    solver_parameters_setting = solver_setting_prompt.format(solver=method)
    solver_parameters_setting += str(specifications)
    return solver_parameters_setting


class AutoOR(object):
    def __init__(self, llm_key=key, llm_url=url, max_tokens=1024):
        self.llm = ChatOpenAI(model='deepseek-chat', openai_api_key=llm_key, openai_api_base=llm_url, max_tokens=max_tokens)
        self.current_problem_name = None
        self.current_problem_setting = None
        self.current_problem_introduction = None
        self.current_solver = None
        self.current_solver_parameters = None
        self.current_recommend_solver = None
        self.simopt_sandbox = None

    def get_problem(self, problem_name="Min Deterministic Function + Noise (SUCG)"):
        self.current_problem_name = problem_name
        self.current_problem_introduction = get_problem_introduction(self.current_problem_name)
        self.current_problem_setting = get_problem_setting(self.current_problem_name)

    def get_solver(self, method="ASTRO-DF (SBCN)"):
        self.current_solver = method
        self.current_solver_parameters = get_methods_parameters(method=self.current_solver)

    def build_simopt_sandbox(self):
        if self.current_problem_name is None or self.current_solver is None:
            print("Please set both problem and solver before building the sandbox.")
            return

        problem_class = problem_unabbreviated_directory[self.current_problem_name]
        solver_class = solver_unabbreviated_directory[self.current_solver]

        problem_instance = problem_class()
        solver_instance = solver_class()

        self.simopt_sandbox = ProblemSolver(problem_instance, solver_instance)

    def autosetting(self, step_nums=100):
        for i in range(step_nums):
            # Step 1: Choose OR optimizer method
            self.current_problem_introduction, _ = get_problem_introduction(self.current_problem_name)
            methods_intro = get_methods_list_and_introduction()
            prompt = (self.current_problem_introduction + "\n" + self.current_problem_setting + "\n" + methods_intro)
            response = self.llm(prompt).strip()
            recommended_solver = response

            if recommended_solver not in solver_introduction_directory:
                print(f"Step {i + 1}: The recommended solver is not valid.")
                continue

            self.get_solver(recommended_solver)

            # Step 2: Choose OR optimizer parameter
            parameters_prompt = (self.current_problem_introduction + "\n" + self.current_problem_setting + "\n" +
                                 f"Chosen solver: {recommended_solver}\n" + get_methods_parameters(recommended_solver))
            solver_parameters_response = self.llm(parameters_prompt).strip()
            self.current_solver_parameters = solver_parameters_response

            # Step 3: Run the method with chosen parameter in the sandbox
            self.build_simopt_sandbox()
            result = self.simopt_sandbox.solve(
                parameters=self.current_solver_parameters)  # Assuming solve method accepts parameters
            print(f"Step {i + 1} result: ", result)

            # Step 4: Based on the result, update the method and parameter
            update_prompt = (self.current_problem_introduction + "\n" + self.current_problem_setting + "\n" +
                             f"Result: {result}\n" + "Suggest updates to the solver method and parameters.")
            update_response = self.llm(update_prompt).strip()
            updated_solver, updated_parameters = update_response.split(
                ';')  # Assuming the response is "solver;parameters"

            if updated_solver in solver_introduction_directory:
                self.current_solver = updated_solver
                self.current_solver_parameters = updated_parameters
            else:
                print(f"Step {i + 1}: The updated solver is not valid.")

    def autotraining(self, step_nums=100):
        for i in range(step_nums):
            # Step 1: Return the OR method code
            self.current_problem_introduction, _ = get_problem_introduction(self.current_problem_name)
            methods_intro = get_methods_list_and_introduction()
            prompt = (self.current_problem_introduction + "\n" + self.current_problem_setting + "\n" + methods_intro)
            response = self.llm(prompt).strip()
            recommended_solver = response

            if recommended_solver not in solver_introduction_directory:
                print(f"Step {i + 1}: The recommended solver is not valid.")
                continue

            self.get_solver(recommended_solver)

            # Step 2: Choose OR optimizer parameter
            parameters_prompt = (self.current_problem_introduction + "\n" + self.current_problem_setting + "\n" +
                                 f"Chosen solver: {recommended_solver}\n" + get_methods_parameters(recommended_solver))
            solver_parameters_response = self.llm(parameters_prompt).strip()
            self.current_solver_parameters = solver_parameters_response

            # Step 3: Run the method with chosen parameter in the sandbox
            self.build_simopt_sandbox()
            result = self.simopt_sandbox.solve(
                parameters=self.current_solver_parameters)  # Assuming solve method accepts parameters
            print(f"Step {i + 1} result: ", result)

            # Step 4: Based on the result, update the method code and parameter
            update_prompt = (self.current_problem_introduction + "\n" + self.current_problem_setting + "\n" +
                             f"Result: {result}\n" + "Suggest updates to the solver method code and parameters.")
            update_response = self.llm(update_prompt).strip()
            updated_solver, updated_parameters = update_response.split(
                ';')  # Assuming the response is "solver_code;parameters"

            if updated_solver in solver_introduction_directory:
                self.current_solver = updated_solver
                self.current_solver_parameters = updated_parameters
            else:
                print(f"Step {i + 1}: The updated solver code is not valid.")
