import itertools
import os.path
from multiprocessing import Pool
from random import seed, random
from collections import OrderedDict
import sys
import hashlib
import json
import pandas as pd
from maze_ga import MazeGa
from numpy import linspace
import time

OVERWRITE_ON_CHANGE = False
SAVE_FREQUENCY = 1
SOLVING_CHUNK_SIZE = 1
MAX_PROCESSES = 10

class Storage():
    def __init__(self, run_file, generation_file, settings_file, save_frequency=1):
        self.run_file = run_file
        self.generation_file = generation_file
        self.settings_file = settings_file
        self.last_run_id = -1
        self.save_frequency = save_frequency
        self.save_requests = 0
        self.settings_hash = ""

    def load_header(self):
        self.run_df = pd.read_parquet(self.run_file, engine='fastparquet')
        self.generation_df = pd.read_parquet(self.generation_file, engine='fastparquet')

    def set_header(self, runs_header, generations_header):
        self.run_df = pd.DataFrame(columns=runs_header)
        self.generation_df = pd.DataFrame(columns=generations_header)
        self.store_runs(append=False)
        self.store_generations(append=False)

    def check_header(self, runs_header, generations_header):
        if not self.storage_ok():
            self.set_header(runs_header, generations_header)
        else:
            self.load_header()
            if not set(runs_header) == set(self.run_df.columns):
                raise ValueError('Run header does not match')
            if not set(generations_header) == set(self.generation_df.columns):
                raise ValueError('Generation header does not match')
        
    def storage_ok(self):
        return os.path.isfile(self.run_file) and os.path.isfile(self.generation_file) and os.path.isfile(self.settings_file)
    
    def set_settings_hash(self, settings_hash):
        self.settings_hash = settings_hash

    def load_settings(self):
        with open(self.settings_file, 'r') as f:
            try:
                self.settings_hash = f.readline().strip("\n")
                run_id = f.readline().strip("\n")
                if run_id != "":
                    self.last_run_id = int(run_id)
            except ValueError:
                pass

    def update_settings(self):
        with open(self.settings_file, 'w') as f:
            f.write(self.settings_hash)
            f.write("\n")
            f.write(str(self.last_run_id))
            f.write("\n")
    
    def store_runs(self, append=True):
        self.last_run_id = self.run_df["run_id"].max()
        self.update_settings()
        self.run_df.to_parquet(self.run_file, engine='pyarrow', compression='gzip', append=append)
    
    def store_generations(self, append=True):
        self.generation_df.to_parquet(self.generation_file, engine='pyarrow', compression='gzip', append=append)

    def concat_runs(self, run_result):
        run_result_df = pd.DataFrame(run_result, index=[0])
        self.run_df = pd.concat([self.run_df, run_result_df], ignore_index=True)

    def concat_generations(self, generation_result):
        generation_result_df = pd.DataFrame(generation_result)
        self.generation_df = pd.concat([self.generation_df, generation_result_df], ignore_index=True)
        
    def save(self, results):
        for parameters, solutions in results:
            # Store run to datraframe
            self.concat_runs(parameters)
            # Store each generation to dataframe
            for i in range(len(solutions)):
                solution_fitness, solution_path = solutions[i]
                generation_result = {
                    "run_id": parameters["run_id"],
                    "generation": i,
                    "fitness": solution_fitness,
                    "path": solution_path
                }
                self.concat_generations(generation_result)
            
            # Save to disk if save frequency is reached
            self.save_requests += 1
            if self.save_requests >= self.save_frequency:
                self.store_runs()
                self.store_generations()
                self.save_requests = 0
        

class Analysis():
    parameters = OrderedDict({
        "maze_file": ["maze_1.txt", "maze_2.txt", "maze_3.txt", "maze_4.txt", "maze_5.txt", "maze_treasure_2.txt", "maze_treasure_3.txt", "maze_treasure_4.txt", "maze_treasure_5.txt"],
        "generations": [300],
        "valid_only": [True, False],
        "population_size": list(range(50, 500, 50)),
        "parents": linspace(0.01, 0.25, 5),
        "mutation_probability": linspace(0.01, 0.1, 5),
        "elitism": linspace(0.01, 0.05, 5),
        "custom_mutation" : [True, False],
        "custom_crossover" : [True, False],
    })
    mazes = {}

    def load_mazes(self, maze_dir):
        self.mazes = {}
        for maze_file in self.parameters["maze_file"]:
            with open(os.path.join(maze_dir, maze_file), "r") as f:
                self.mazes[maze_file] = f.read()
    
    def combinations(self):
        named_parameter_cominations = [dict(zip(self.parameters.keys(), values)) for values in itertools.product(*self.parameters.values())]
        print("Number of combinations: {}".format(len(named_parameter_cominations)))
        print("Example combination: {}".format(named_parameter_cominations[0]))
        return named_parameter_cominations
    
    def parameters_hash(self):
        dhash = hashlib.md5()
        for key, value in self.parameters.items():
            dhash.update(str(key).encode('utf-8'))
            dhash.update(str(value).encode('utf-8'))
        return dhash.hexdigest()
   
def execute_combination(run_id, parameters):
    # extract maze file path and load it
    maze_ga = MazeGa(
        parameters["maze_file"], 
        custom_mutation=parameters["custom_mutation"],
        custom_crossover=parameters["custom_crossover"], 
        valid_only=parameters["valid_only"], 
        show_each_n=0, 
        threads=5
    )

    # execute ga algorithem and retrive results
    solutions_fintness, best_solutions = maze_ga.run(
            parameters["generations"], 
            parameters["population_size"], 
            parameters["parents"], 
            parameters["mutation_probability"], 
            parameters["elitism"]
    )

    solutions = list(zip(solutions_fintness, best_solutions))
    parameters["run_id"] = run_id

    print(run_id, "finished")
    return parameters, solutions

if __name__ == "__main__":
    # read args
    if len(sys.argv) > 1:
        OVERWRITE_ON_CHANGE = sys.argv[1] == "-ns"

    # set random seed
    seed(100)

    # expected header of runs file
    runs_header = ["run_id", "start_time", "end_time", "maze_file", "valid_only", "population_size", "parents", "mutation_probability", "elitism", "custom_mutation", "custom_crossover"]
    # expected header of generations file
    generations_header = ["run_id", "start_time", "end_time", "generation", "fitness", "solution"]

    # storage
    storage = Storage("./analysis/runs.parquet .", "./analysis/generations.parquet .", "./analysis/settings.txt", save_frequency=SAVE_FREQUENCY)
    storage.check_header(runs_header, generations_header)
    storage.load_settings()

    # analysis parameters
    analysis = Analysis()
    analysis.load_mazes("./mazes")
    combinations = analysis.combinations()
    settings_hash = analysis.parameters_hash()

    # check if settings have changed
    if storage.settings_hash != settings_hash:
        print("Settings have changed")
        if OVERWRITE_ON_CHANGE:
            print("Overwriting old settings and data (10s)")
            time.sleep(10)
            storage.set_header(runs_header, generations_header)
            storage.set_settings_hash(settings_hash)
            storage.update_settings()
        else:
            exit(1)

    # display data about last run and remaining combinations
    print("Last run id: {}".format(storage.last_run_id))
    enumerated_combinations = list(enumerate(combinations))
    combinations = enumerated_combinations[storage.last_run_id+1:]
    print("Number of combinations to execute: {}".format(len(combinations)))

    # start thread pool
    with Pool(processes=MAX_PROCESSES) as pool:
        for i in range(0, len(combinations), SOLVING_CHUNK_SIZE):
            solving_chunk = combinations[i:i + SOLVING_CHUNK_SIZE]
            # convert maze names to maze content
            for i in range(len(solving_chunk)):
                solving_chunk[i][1]["maze_file"] = analysis.mazes[solving_chunk[i][1]["maze_file"]]
            # evaluate chunk
            results = pool.starmap(execute_combination, solving_chunk)
            storage.save(results)
        pool.close()
        pool.join()

    # final save
    storage.store_runs()
    storage.store_generations()