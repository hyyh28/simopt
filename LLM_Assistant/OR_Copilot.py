from langchain_openai import ChatOpenAI
from LLM import url, key
from simopt.directory import problem_directory, model_introduction_directory, problem_unabbreviated_directory, solver_directory, solver_unabbreviated_directory, model_directory, model_unabbreviated_directory, model_problem_unabbreviated_directory, model_problem_class_directory
from simopt.experiment_base import ProblemSolver, ProblemsSolvers, post_normalize, find_missing_experiments, make_full_metaexperiment, plot_progress_curves, plot_solvability_cdfs, plot_area_scatterplots, plot_solvability_profiles, plot_terminal_progress, plot_terminal_scatterplots

llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=key,
    openai_api_base=url,
    max_tokens=1024
)

from langchain_core.prompts import PromptTemplate
user_prompt_of_introduction = PromptTemplate.from_template("You choose {problem} based on {model} model, which is {model_introduction}")
llm_prompt_of_introduction = PromptTemplate.from_template("The user choose {problem} based on {model} model, which is {model_introduction}")

def get_problem_introduction(problem="Min Deterministic Function + Noise (SUCG)"):
    model_of_problem = model_problem_unabbreviated_directory[problem]
    introduction_of_model = model_introduction_directory[model_of_problem]
    llm_problem_introduction = llm_prompt_of_introduction.format(problem=problem, model=model_of_problem,
                                                                 model_introduction=introduction_of_model)
    user_problem_introduction = llm_prompt_of_introduction.format(problem=problem, model=model_of_problem,
                                                                 model_introduction=introduction_of_model)
    return llm_problem_introduction, user_problem_introduction

def get_problem_setting(problem="Min Deterministic Function + Noise (SUCG)"):





print(problem_unabbreviated_directory.keys())
print(problem_directory.keys())