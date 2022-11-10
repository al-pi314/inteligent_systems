import itertools
from random import seed
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Lock
import pandas as pd
from maze_ga import MazeGa
from numpy import arange

lock = Lock()

def encode_path(path):
    # convert path array into a string that can be later used for creating an agent
    return ''.join([str(x) for x in path])

def save_to_csv():
    print("saving to file. Num of writes:", writes)
    # stores data from data frames to file on disk (for safety so that the results are not lost in case of an error)
    runs.to_csv(save_dir + "runs.csv", index=True)
    generations_scores.to_csv(save_dir + "generations_scores.csv", index=True)

def save_to_df(data):
    # lock access to shared resources
    lock.acquire()
    global writes
    for result in data:
        # unpack result data
        solutions_fintness, best_solutions, params, run_id = result
        # add each solution fitness in order to generation_scores data frame
        for i in range(len(solutions_fintness)):
            generations_scores.loc[-1] = [run_id, i, solutions_fintness[i], encode_path(best_solutions[i])]
            generations_scores.index = generations_scores.index + 1

        # add run data to runs data frame + add run_id that can be mapped to generations_scores data frame
        runs.loc[-1] = list(params) + [run_id]
        runs.index = runs.index + 1

        # check if the save should also update the actual file on disk
        if writes % save_on_n_writes == 0:
            # update file on disk
            save_to_csv()
        writes += 1

    # unlock resources
    lock.release()
        

def execute_combinations(combinations, start_run_id):
    # data stores all calculated results
    data = []
    for run_id, params in enumerate(combinations):
        # extract maze file path and load it
        maze_file = params[0]
        maze_ga = MazeGa()
        maze_ga.encode_maze(open(directory + maze_file, "r").read())

        # other parameters are needed for ga function call + display = False parameter
        func_params = list(params[1:]) + [False]
        # execute ga algorithem and retrive results
        solutions_fintness, best_solutions = maze_ga.run_ga(*func_params)
        # store results for later use
        data.append((solutions_fintness, best_solutions, params, start_run_id + run_id))

    # at the end write all data to data frame (uses locking)
    save_to_df(data) 


if __name__ == "__main__":
    # set random seed
    seed(100)

    # directories paths (must exist)
    directory = "./mazes/"
    save_dir = "/home/kerikon/Development/inteligent_systems/seminar_1/analysis/"

    # all possible parameters
    parameters = {
        "maze_file": ["maze_1.txt", "maze_2.txt", "maze_3.txt", "maze_4.txt", "maze_5.txt", "maze_6.txt", "maze_7.txt", "maze_treasure_2.txt", "maze_treasure_3.txt", "maze_treasure_4.txt", "maze_treasure_5.txt", "maze_treasure_6.txt", "maze_treasure_7.txt"],
        "generations": [500],
        "custom_population_func": [True, False],
        "population_size": list(range(25, 251, 25)),
        "parents": arange(0.02, 0.53, 0.05),
        "mutation_rate": arange(0.05, 0.51, 0.05),
        "elitism": arange(0, 0.051, 0.01),
        "custom_functions": [True, False],
    }

    # every possible combination of parameters
    combinations = list(itertools.product(*parameters.values()))
    combinations_len = len(combinations)
    print("combinations size", combinations_len)

    # data frames that store all run results
    runs = pd.DataFrame(columns=list(parameters.keys()) + ["run"])
    generations_scores = pd.DataFrame(columns=["run", "generation", "score", "path"])

    # how frequently to write to file
    save_on_n_writes = 3000
    writes = 0

    # thread pool parameters
    n_theads = 10
    task_size = 5
    task_idx = 0

    # start thread pool
    with ThreadPoolExecutor(max_workers=n_theads) as executor:
        while task_size * task_idx <= combinations_len:
            # combinations that will be evaluated by a thread
            thread_combinations = combinations[task_idx * task_size: ((task_idx +1) * task_size)]
            # start thread with parameters
            executor.submit(execute_combinations, thread_combinations, task_size * task_idx)
            # update task status
            task_idx += 1
    # final save
    save_to_csv()