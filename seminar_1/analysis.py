import itertools
from random import seed
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Lock
import pandas as pd
from maze_ga import MazeGa
from numpy import arange, linspace
from time import time as seconds

generations_scores_dataframe_lock = Lock()
runs_dataframe_lock = Lock()
generations_scores_filewrite_lock = Lock()
runs_filewrite_lock = Lock()
global_write_lock = Lock()

def encode_path(path):
    # convert path array into a string that can be later used for creating an agent
    return ''.join([str(x) for x in path])

def save_to_csv():
    # save runs data to csv
    runs_filewrite_lock.acquire()
    runs.to_csv(save_dir + "runs.csv", index=True)
    runs_filewrite_lock.release()

    # save generations_scores to csv
    generations_scores_filewrite_lock.acquire()
    generations_scores.to_csv(save_dir + "generations_scores.csv", index=True)
    generations_scores_filewrite_lock.release()

def save_to_df(solutions_fintness, best_solutions, combination, run_id):
    # add each solution fitness in order to generation_scores data frame
    generations_scores_dataframe_lock.acquire()
    for i in range(len(solutions_fintness)):
        generations_scores.loc[-1] = [run_id, i, solutions_fintness[i], encode_path(best_solutions[i])]
        generations_scores.index = generations_scores.index + 1
    generations_scores_dataframe_lock.release()


    # add run data to runs data frame + add run_id that can be mapped to generations_scores data frame
    runs_dataframe_lock.acquire()
    runs.loc[-1] = list(combination) + [run_id]
    runs.index = runs.index + 1
    runs_dataframe_lock.release()

    # check writes
    if run_id % save_on_n_runs == 0:
        print("time needed per task", (round(seconds() * 1000) - start_time) / run_id)
        save_to_csv()

def execute_combination(input_data):
    run_id, combination = input_data

    # extract maze file path and load it
    maze_file = combination[0]
    maze_ga = MazeGa()
    maze_ga.encode_maze(file_data[maze_file])

    # other parameters are needed for ga function call + display = False parameter
    func_params = list(combination[1:]) + [False]
    # execute ga algorithem and retrive results
    solutions_fintness, best_solutions = maze_ga.run_ga(*func_params)

    # at the end write all data to data frame (uses locking)
    print(run_id, "finished")
    save_to_df(solutions_fintness, best_solutions, combination, run_id) 


if __name__ == "__main__":
    # set random seed
    seed(100)

    # directories paths (must exist)
    directory = "./mazes/"
    save_dir = "/home/kerikon/Development/inteligent_systems/seminar_1/analysis/"

    # all possible parameters
    parameters = {
        "maze_file": ["maze_1.txt", "maze_2.txt", "maze_3.txt", "maze_4.txt", "maze_5.txt", "maze_6.txt", "maze_7.txt", "maze_treasure_2.txt", "maze_treasure_3.txt", "maze_treasure_4.txt", "maze_treasure_5.txt", "maze_treasure_7.txt"],
        "generations": [150],
        "custom_population_func": [True, False],
        "population_size": list(range(25, 251, 25)),
        "parents": linspace(0.01, 0.25, 5),
        "mutation_rate": linspace(0.01, 0.1, 5),
        "elitism": [0, 0.01, 0.1],
        "custom_functions": [True, False],
    }

    file_data = {}
    for name in parameters["maze_file"]:
        f = open(directory + name, "r")
        file_data[name] = f.read()
        f.close()

    # every possible combination of parameters
    combinations = list(itertools.product(*parameters.values()))
    combinations_len = len(combinations)
    print("combinations size", combinations_len)

    # data frames that store all run results
    runs = pd.DataFrame(columns=list(parameters.keys()) + ["run"])
    generations_scores = pd.DataFrame(columns=["run", "generation", "score", "path"])

    # start time
    start_time = round(seconds() * 1000)

    # save on n runs
    save_on_n_runs = 100

    # thread pool parameters
    max_workers = 500

    # start thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for future in executor.map(execute_combination, enumerate(combinations)):
            future.results()
        

    # final save
    save_to_csv()