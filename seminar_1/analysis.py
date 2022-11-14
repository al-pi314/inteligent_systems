import itertools
import os.path
from multiprocessing import Pool
from random import seed, random

import pandas as pd
from maze_ga import MazeGa
from numpy import linspace

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def encode_path(path):
    # convert path array into a string that can be later used for creating an agent
    return ''.join([str(x) for x in path])

def error_callback(e):
    print("ERROR: ", e)

def save_to_csv():
    print("saving to csv")

    # save runs data to csv
    runs.to_csv(analysis_runs_file, index=True)

    # save generations_scores to csv
    generations_scores.to_csv(analysis_generations_scores_file, index=True)

def save_to_df(solutions_fintness, best_solutions, parameters, run_id):
    global runs, generations_scores
    # add each solution fitness in order to generation_scores data frame
    generation_stats = []
    for i in range(len(solutions_fintness)):
        generation_stats.append([run_id, i, solutions_fintness[i], encode_path(best_solutions[i])])
    generations_scores = generations_scores.append(pd.DataFrame(generation_stats, columns=generations_scores_columns), ignore_index=True)

    # add run data to runs data frame + add run_id that can be mapped to generations_scores data frame
    parameters["run_id"] = run_id
    runs = runs.append(parameters, ignore_index=True)

def save_results(results):
    # save results to data frame
    for solutions_fintness, best_solutions, parameters, run_id in results:
        save_to_df(solutions_fintness, best_solutions, parameters, run_id)
    save_to_csv()

def execute_combination(run_id, parameters, maze_strings):
    # extract maze file path and load it
    maze_string = maze_strings[parameters["maze_file"]]
    maze_ga = MazeGa(maze_string, use_custom_functions=parameters["use_custom_functions"], valid_only=parameters["valid_only"], show_each_n=0, threads=5)

    # execute ga algorithem and retrive results
    solutions_fintness, best_solutions = maze_ga.run(parameters["generations"], parameters["population_size"], parameters["parents"], parameters["mutation_probability"], parameters["elitism"])

    # at the end write all data to data frame (uses locking)
    print(run_id, "finished")
    return solutions_fintness, best_solutions, parameters, run_id

if __name__ == "__main__":
    # set random seed
    seed(100)

    # files
    analysis_runs_file = "./analysis/runs.csv"
    analysis_generations_scores_file = "./analysis/generations_scores.csv"
    maze_directory = "./mazes/"

    # all possible parameters
    parameters = {
        "maze_file": ["maze_1.txt", "maze_2.txt", "maze_3.txt", "maze_4.txt", "maze_5.txt", "maze_6.txt", "maze_7.txt", "maze_treasure_2.txt", "maze_treasure_3.txt", "maze_treasure_4.txt", "maze_treasure_5.txt", "maze_treasure_7.txt"],
        "generations": [150],
        "valid_only": [True, False],
        "population_size": list(range(25, 251, 25)),
        "parents": linspace(0.01, 0.25, 5),
        "mutation_probability": linspace(0.01, 0.1, 5),
        "elitism": [0, 0.01, 0.1],
        "use_custom_functions": [True, False],
    }
    runs_columns = ["maze_file", "generations", "valid_only", "population_size", "parents", "mutation_probability", "elitism", "use_custom_functions", "run_id"]
    generations_scores_columns = ["run_id", "generation", "fitness", "path"]

    runs = pd.DataFrame(columns=runs_columns)
    generations_scores = pd.DataFrame(columns=generations_scores_columns)

    if os.path.isfile(analysis_runs_file):
        runs = pd.read_csv(analysis_runs_file, index_col=0)
        if set(runs_columns) != set(runs.columns):
            print("ERROR: runs file columns missmatch")
            print("expected: ", runs_columns)
            print("got: ", runs.columns)
            exit(1)
    
    if os.path.isfile(analysis_generations_scores_file):
        generations_scores = pd.read_csv(analysis_generations_scores_file, index_col=0)
        if set(generations_scores_columns) != set(generations_scores.columns):
            print("ERROR: generations_scores file columns missmatch")
            print("expected: ", generations_scores_columns)
            print("got: ", generations_scores.columns)
            exit(1)

    # read maze strings
    maze_strings = {}
    for maze_file in parameters["maze_file"]:
        with open(os.path.join(maze_directory, maze_file), "r") as file:
            maze_strings[maze_file] = file.read()


    # every possible combination of parameters
    values_combinations = list(itertools.product(*parameters.values()))
    combinations = [
        dict(zip(parameters.keys(), combination)) for combination in values_combinations
    ]

    combinations_len = len(combinations)
    print(combinations[0])
    print("combinations size", combinations_len)

    # save on n runs
    save_on_n_runs = 10

    # thread pool parameters
    max_processes = 10

    # start thread pool
    with Pool(processes=max_processes) as pool:
        solved = set(runs["run_id"].unique())
        arguments = [(i, combinations[i], maze_strings) for i in range(len(combinations)) if i not in solved]
        print("unsolved", len(arguments))
        for i in range(0, len(arguments), save_on_n_runs):
            sub_arguments = arguments[i:i + save_on_n_runs]
            results = pool.starmap(execute_combination, sub_arguments)
            save_results(results)
        pool.close()
        pool.join()

    # final save
    save_to_csv()