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

class ORCopilot(object):
    def __init__(self, llm_key=key, llm_url=url, max_tokens=1024):
        self.llm = ChatOpenAI(model='deepseek-chat', openai_api_key=key, openai_api_base=url, max_tokens=max_tokens)
        self.current_problem_name = None
        self.current_problem_setting = None
        self.current_problem_introduction = None
        self.current_solver = None
        self.current_recommand_solver = None

    def get_problem(self, problem_name="Min Deterministic Function + Noise (SUCG)"):
        self.current_problem_name = problem_name
        self.current_problem_introduction = get_problem_introduction(self.current_problem_name)
        self.current_problem_setting = get_problem_setting(self.current_problem_name)

    def get_solver(self, method="ASTRO-DF (SBCN)"):
        self.current_solver = method

    def give_recommand_solver(self):
        pass

    def give_recommand_solver_parameter(self):
        pass



