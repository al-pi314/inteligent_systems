from itertools import product
from os import path, listdir, makedirs
from multiprocessing import Pool
from collections import OrderedDict
from hashlib import md5
import pandas as pd
from maze_ga import MazeGa
from numpy import linspace

SAVE_FREQUENCY = 3
SOLVING_CHUNK_SIZE = 100
MAX_PROCESSES = 15

class Storage():
    def __init__(self, runs_dir, generations_dir, settings_file, runs_schema, runs_index, generations_schema, generations_index, save_frequency=1):
        self.runs_dir = runs_dir
        self.generations_dir = generations_dir
        self.settings_file = settings_file

        makedirs(self.runs_dir, exist_ok=True)
        makedirs(self.generations_dir, exist_ok=True)

        self.saves = 0

        self.save_frequency = save_frequency
        self.save_requests = 0

        self.runs_schema = runs_schema
        self.runs_index = runs_index
        self.generations_schema = generations_schema
        self.generations_index = generations_index

        self.run_df = None
        self.generation_df = None

        self.settings_hash = ""
        self.last_run_id = -1

    def load_from_disk(self, settings_hash):
        ids = self.save_ids()
        if path.isfile(self.settings_file):
            sh, lri = self.load_settings()
            schema_valid, run_ids = self.read_schema()
            if sh == settings_hash and schema_valid:
                self.saves = max(ids) +1
                self.settings_hash = sh
                self.last_run_id = lri
                print("continue from last run")
            else:
                if sh != settings_hash:
                    print("settings changed")
                else:
                    print("schema changed")
                self.saves = 0
                self.settings_hash = settings_hash
                self.last_run_id = -1
                return set(), set()
            return ids, run_ids
        
        print("start new run")
        self.update_settings()
        return ids, run_ids 
        
    def file_path(self, file_dir):
        return path.join(file_dir, f'file_{self.saves}.parquet')

    def load_settings(self):
        sh, lri = "", -1
        with open(self.settings_file, 'r') as f:
            settings = [l.strip("\n") for l in f.readlines()]
            if len(settings) >= 2:
                sh = settings[0]
                lri = int(settings[1])
        return sh, lri

    def update_settings(self):
        with open(self.settings_file, 'w') as f:
            f.write(self.settings_hash)
            f.write("\n")
            f.write(str(self.last_run_id))
            f.write("\n")

    def read_schema(self):
        run_df = pd.read_parquet(self.runs_dir, engine='pyarrow')
        generation_df = pd.read_parquet(self.generations_dir, engine='pyarrow')

        run_ids = set(run_df.index)

        must_be_equal = [(run_df.columns, self.runs_schema), (generation_df.columns, self.generations_schema), (run_df.index.names, self.runs_index), (generation_df.index.names, self.generations_index)]
        print([set(a) - set(b) for a, b in must_be_equal])
        print([set(b) - set(a) for a, b in must_be_equal])
        return all([set(a) == set(b) for a, b in must_be_equal]), run_ids

    def save_ids(self):
        ids = set()
        for filename in listdir(self.runs_dir):
            if filename.endswith(".parquet"):
                save_id = int(filename.split("_")[1].split(".")[0])
                ids.add(save_id)
        return ids

    def store_to_df(self, run_df, generation_df):
        if self.run_df is None:
            self.run_df = run_df
        else:
            self.run_df = pd.concat([self.run_df, run_df])

        if self.generation_df is None:
            self.generation_df = generation_df
        else:
            self.generation_df = pd.concat([self.generation_df, generation_df])
        print("stored to DF")

    def store_to_disk(self):
        self.generation_df.to_parquet(self.file_path(self.generations_dir), engine='pyarrow', compression='gzip', index=True)
        self.run_df.to_parquet(self.file_path(self.runs_dir), engine='pyarrow', compression='gzip', index=True)
        
        self.update_settings()

        self.run_df = None
        self.generation_df = None
        self.saves += 1
        print("stored to DISK")
        
    def save(self, results):
        runs_dicts = []
        generations_dicts = []
        for parameters, solutions in results:
            # Store run to datraframe
            runs_dicts.append(parameters)

            # Store each generation to dataframe
            for i in range(len(solutions)):
                solution_fitness, solution_path = solutions[i]
                generation_result = {
                    "run_id": parameters["run_id"],
                    "start_time": None,
                    "end_time": None,
                    "generation": i,
                    "fitness": solution_fitness,
                    "solution": ''.join(map(str, solution_path)),
                }
                generations_dicts.append(generation_result)

            # update last run id
            if parameters["run_id"] > self.last_run_id:
                self.last_run_id = parameters["run_id"]
        
        # store to dataframe
        run_df = pd.DataFrame(runs_dicts)
        run_df.set_index(self.runs_index, inplace=True)

        generation_df = pd.DataFrame(generations_dicts)
        generation_df.set_index(self.generations_index, inplace=True)

        self.store_to_df(run_df, generation_df)
            
        # Save to disk if save frequency is reached
        self.save_requests += 1
        if self.save_requests >= self.save_frequency:
            self.store_to_disk()
            self.save_requests = 0
        

class Analysis():
    parameters = OrderedDict({
        "maze_file": ["maze_1.txt", "maze_2.txt", "maze_3.txt", "maze_4.txt", "maze_5.txt", "maze_treasure_2.txt", "maze_treasure_3.txt", "maze_treasure_4.txt", "maze_treasure_5.txt"],
        "generations": [300],
        "valid_only": [True, False],
        "population_size": list(range(50, 550, 50)),
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
            with open(path.join(maze_dir, maze_file), "r") as f:
                self.mazes[maze_file] = f.read()
    
    def combinations(self):
        named_parameter_cominations = [dict(zip(self.parameters.keys(), values)) for values in product(*self.parameters.values())]
        print("Number of combinations: {}".format(len(named_parameter_cominations)))
        print("Example combination: {}".format(named_parameter_cominations[0]))
        return named_parameter_cominations
    
    def parameters_hash(self):
        dhash = md5()
        for key, value in self.parameters.items():
            dhash.update(str(key).encode('utf-8'))
            dhash.update(str(value).encode('utf-8'))
        return dhash.hexdigest()
   
def execute_combination(run_id, parameters, maze_string):
    # extract maze file path and load it
    maze_ga = MazeGa(
        maze_string, 
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
    parameters["start_time"] = None
    parameters["end_time"] = None

    print(run_id, "finished")
    return parameters, solutions

if __name__ == "__main__":
    # expected schema of runs file
    runs_schema = ["start_time", "end_time", "maze_file", "generations", "valid_only", "population_size", "parents", "mutation_probability", "elitism", "custom_mutation", "custom_crossover"]
    runs_index = ["run_id"]
    # expected schema of generations file
    generations_schema = ["start_time", "end_time", "fitness", "solution"]
    generations_index = ["run_id", "generation"]

    # analysis parameters
    analysis = Analysis()
    analysis.load_mazes("./mazes")
    combinations = analysis.combinations()

    # storage
    storage = Storage("./analysis/test_run/runs", "./analysis/test_run/generations", "./analysis/test_run/settings.txt", 
        runs_schema, runs_index, generations_schema, generations_index, 
        save_frequency=SAVE_FREQUENCY)

    # load previous runs
    save_ids, run_ids = storage.load_from_disk(analysis.parameters_hash())

    # display data about last run and remaining combinations
    print("Last run id: {}".format(storage.last_run_id))
    combinations = [(i, combination, analysis.mazes[combination["maze_file"]]) for i, combination in enumerate(combinations) if i not in run_ids]
    print("Number of combinations to execute: {}".format(len(combinations)))

    # wait for user input
    print("VALIDATE ALL SETTINGS ... PRESS ANY KEY TO CONTINUTE")
    input()
    print("starting ...")

    # start thread pool
    with Pool(processes=MAX_PROCESSES) as pool:
        for i in range(0, len(combinations), SOLVING_CHUNK_SIZE):
            solving_chunk = combinations[i:i + SOLVING_CHUNK_SIZE]
            # evaluate chunk
            results = pool.starmap(execute_combination, solving_chunk)
            storage.save(results)
        pool.close()
        pool.join()

    # final save
    storage.store_runs()
    storage.store_generations()