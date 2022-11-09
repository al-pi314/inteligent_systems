import itertools
from random import seed
from threading import Lock

lock = Lock()

import pandas as pd
from maze_ga import encode_maze, run_ga
from numpy import arange


def encode_path(path):
    return ''.join([str(x) for x in path])

def save_to_df(data):
    lock.acquire()
    for d in data:
        solutions_fintness, best_solutions, params, run_id = d
        for i in range(len(solutions_fintness)):
            generations_scores.loc[-1] = [run_id, i, solutions_fintness[i], encode_path(best_solutions[i])]
            generations_scores.index = generations_scores.index + 1
        save_params = list(params) + [run_id]
        runs.loc[-1] = save_params
        # print(save_params)
        runs.index = runs.index + 1

        writes += 1
        if writes % save_on_n_threads_finish == 0:
            runs.to_csv("runs.csv", index=True)
            generations_scores.to_csv("generations_scores.csv", index=True)
    lock.release()
        

def run_combinations(combinations, start_run_id):
    data = []
    for run_id, params in enumerate(combinations):
        run_id += start_run_id
        # print(params)
        maze_file = params[0]
        encode_maze(open(directory + maze_file, "r").read())

        c = list(params[1:])
        func_params = c + [False] # disable display
        solutions_fintness, best_solutions = run_ga(*func_params)
        # print(solutions_fintness)
        data.append((solutions_fintness, best_solutions, params, run_id))
    save_to_df(data)   

if __name__ == "__main__":
    seed(100)

    directory = "./mazes/"
    parameters = {
        "maze_file": ["maze_1.txt", "maze_2.txt", "maze_3.txt", "maze_4.txt", "maze_5.txt", "maze_6.txt", "maze_7.txt", "maze_treasure_2.txt", "maze_treasure_3.txt", "maze_treasure_4.txt", "maze_treasure_5.txt", "maze_treasure_6.txt", "maze_treasure_7.txt"],
        "generations": list(range(25, 501, 25)),
        "custom_population_func": [True, False],
        "population_size": list(range(25, 251, 25)),
        "parents": arange(0.02, 0.53, 0.05),
        "mutation_rate": arange(0.05, 0.51, 0.05),
        "elitism": arange(0, 0.051, 0.01),
        "custom_mutation_func": [True, False],
        "custom_crossover_func": [True, False],
    }

    combinations = list(itertools.product(*parameters.values()))
    print(combinations[0], combinations[-1])
            
    runs = pd.DataFrame(columns=list(parameters.keys()) + ["run"])
    generations_scores = pd.DataFrame(columns=["run", "generation", "score", "path"])

    save_on_n_writes = 1
    writes = 0

    threads_n = 1
    per_thread_tasks = 10
    threads
    
    for thread in all_t:
        thread.join()
    
