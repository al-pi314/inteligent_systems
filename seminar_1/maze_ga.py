import json
import numpy as np
import pygad
from sys import argv
from display import on_generation_factory, show_solution
from function_factory import (crossover_factory, fitness_factory,
                              generate_population_factory, mutation_factory)


class MazeGa:
    # global variables
    random_seed = 200
    maze = None
    maze_start = None
    treasures = 0

    # factor to use for penalties & rewards
    TREASURE_MULTIPLIER = 1
    FINISH_MULTPILER = 1
    UNIQUE_MOVE_REWARD = 1
    REPEATED_MOVE_REWARD = 0.9
    BONUS_MOVE_REWARD = 2
    INVALID_MOVE_PENALTY = 2

    # two-way encoding mappings
    encoding = {
        "#": 0,
        ".": 1,
        "S": 2,
        "E": 3,
        "T": 4,
    }
    encoding_reverse = {v: k for k, v in encoding.items()}

    # two-way direction mappings
    directions = {
        "L": 0,
        "R": 1,
        "U": 2,
        "D": 3
    }
    directions_reverse = {v: k for k, v in directions.items()}

    def __init__(self, maze_string, use_custom_functions=False, valid_only=False, show_each_n=0, wait_on_show=False, threads=0):
        self.encode_maze(maze_string)

        self.mutation = "scramble"
        self.crossover = "single_point"
        if use_custom_functions:
            self.mutation = mutation_factory(self)
            self.crossover = crossover_factory(self)


        self.fitness = fitness_factory(self)
        self.generate_population = generate_population_factory(self, valid_only)
        
        self.display = False
        if show_each_n > 0:
            self.display = True
            self.on_generation = on_generation_factory(self, show_each_n, wait_on_show)

        self.multithread = False
        self.threads = 0
        if threads > 0:
            self.multithread = True
            self.threads = threads

    def path_str_to_list(self, path_str):
        return [int(p) for p in path_str]

    def encode_maze(self, maze_string):
        self.treasures = 0

        rows = maze_string.split("\n")
        self.maze = np.zeros((len(rows), len(rows[0])))
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                val = rows[i][j]
                self.maze[i, j] = self.encoding[val]
                if val == "S":
                    self.maze_start = (j, i)
                if val == "T":
                    self.treasures += 1

    def move(self, p, curr):
        rows, columns = self.maze.shape
        n_curr = list(curr)

        # move
        if p == self.directions["L"]:
            n_curr[0] -= 1
        elif p == self.directions["R"]:
            n_curr[0] += 1
        elif p == self.directions["U"]:
            n_curr[1] -=1
        elif p == self.directions["D"]:
            n_curr[1] += 1

        # checks
        # out of bounds
        if n_curr[0] < 0 or n_curr[0] >= columns or n_curr[1] < 0 or n_curr[1]  >= rows:
            return curr, False
        # in wall
        elif self.maze[n_curr[1], n_curr[0]] == self.encoding["#"]:
            return curr, False
        # valid move
        return n_curr, True


    def run(self, generations, population_size, parents, mutation_probability, elitism):
        # initialize population
        initial_population = np.array(self.generate_population(population_size))

        # setup ga algorithem
        ga = pygad.GA(
            # main settings
            random_seed=self.random_seed,
            num_generations=generations,
            num_parents_mating=max(1, int(parents * population_size)),
            parent_selection_type="sus",

            # initial population
            initial_population=initial_population,

            # agent gene type
            gene_type=int,
            init_range_low=0, # lowest valid value for gene (inclusive)
            init_range_high=4, # highest valid value for gene (exclusive)

            # agent evaluation
            mutation_probability=mutation_probability,
            keep_elitism=int(elitism * population_size), # keep best n solutions in the next generation

            # custom functions
            fitness_func=self.fitness,
            mutation_type=self.mutation,
            crossover_type=self.crossover,

            save_best_solutions=True,
            suppress_warnings=True
        )

        if self.multithread:
            ga.parallel_processing = ["threading", self.threads]

        if self.display:
            ga.on_generation = self.on_generation

        print("Example of starting solution 1:")
        show_solution(self, initial_population[10], 0, False)

        print("Example of starting solution 2:")
        show_solution(self, initial_population[90], 0, False)

        print("Example of crossover")
        crossed = maze_ga.crossover([initial_population[10], initial_population[90]], (1, initial_population[0].size), ga)
        show_solution(self, crossed[0], 0, False)

        print("Example of muation")
        mutated = maze_ga.mutation(crossed, ga)
        show_solution(self, mutated[0], 0, False)


        # run multiple tournaments and generations to find the best solution
        ga.run()

        # read & display best solution
        if self.display:
            solution, solution_fitness, _ = ga.best_solution()
            show_solution(self, solution, solution_fitness, sequential_display=True)

        return ga.best_solutions_fitness, ga.best_solutions

if __name__ == "__main__":
    settings_file = "./settings.json"
    if len(argv) > 1:
        settings_file = argv[1]
    with open(settings_file, "r") as f:
        settings = json.load(f)

    print("Running with settings:")
    print(json.dumps(settings, indent=4))


    maze_file = open(settings["maze_file"], "r")
    maze_string = maze_file.read()
    maze_file.close()

    print("Solving maze:")
    print(maze_string)


    maze_ga = MazeGa(maze_string, 
                    use_custom_functions=settings["use_custom_functions"], 
                    valid_only=settings["valid_only"], 
                    show_each_n=settings["show_each_n"], 
                    wait_on_show=settings["wait_on_show"], 
                    threads=settings["threads"])

    maze_ga.run(settings["generations"], 
                settings["population_size"], 
                settings["parents"], 
                settings["mutation_probability"], 
                settings["elitism"])
