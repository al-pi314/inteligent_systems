

from random import choice, random, seed
from sys import argv
from time import sleep

import numpy as np
import pygad

# global variables
RANDOM_SEED = 100
MAZE = None
MAZE_START = None
TREASURES = 0

# factor to use for penalties & rewards
TREASURE_MULTIPLIER = 2
FINISH_MULTPILER = 1
UNIQUE_MOVE_REWARD = 1
REPEATED_MOVE_REWARD = 0.9
BONUS_MOVE_REWARD = 2

# display settings
SHOW_EVERY_N_GENS = 20

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

def encode_maze(s):
    # sets global variables of MAZE and MAZE_START based on encoding values
    global MAZE, MAZE_START, TREASURES, VALID_POINTS

    rows = s.split("\n")
    MAZE = np.zeros((len(rows), len(rows[0])))
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            val = rows[i][j]
            MAZE[i, j] = encoding[val]
            if val == "S":
                MAZE_START = [j, i]
            if val == "T":
                TREASURES += 1

def move(p, curr):
    rows, columns = MAZE.shape
    n_curr = curr[:]

    # move
    if p == directions["L"]:
        n_curr[0] -= 1
    elif p == directions["R"]:
        n_curr[0] += 1
    elif p == directions["U"]:
        n_curr[1] -=1
    elif p == directions["D"]:
        n_curr[1] += 1

    # checks
    # out of bounds
    if n_curr[0] < 0 or n_curr[0] >= columns or n_curr[1] < 0 or n_curr[1]  >= rows:
        return curr, False
    # in wall
    elif MAZE[n_curr[1], n_curr[0]] == encoding["#"]:
        return curr, False
    # valid move
    return n_curr, True


def fitness(path, solution_idx):
    curr = MAZE_START
    visited = set()
    visited.add(tuple(curr))

    collected_treasures = 0
    multiplier = 1
    moves = 0
    found_finish = False
    for i in range(len(path)):
        moves += 1
        p = path[i]

        # check if the move is valid and if any score adjustment is required
        curr, _ = move(p, curr)
        curr_tuple = tuple(curr)
        
        # algorithem found treasure
        if curr_tuple not in visited and MAZE[curr[1], curr[0]] == encoding["T"]:
            collected_treasures += 1
            multiplier += TREASURE_MULTIPLIER

        # algorithem found finish
        if MAZE[curr[1], curr[0]] == encoding["E"]:
            found_finish = True
            multiplier += FINISH_MULTPILER
            break
        
        visited.add(curr_tuple)

    # define move counters
    unique_moves = len(visited)
    repeated_moves = moves - unique_moves
    remaining_moves = len(path) - moves    

    # encurage exploration
    score = unique_moves * UNIQUE_MOVE_REWARD

    # if the maze was solved with all treasures collected
    if found_finish and collected_treasures == TREASURES:
        # we encurage making unique moves, but don't punish repeated ones too much, we also highly encurage using as little moves as possible using bonus reward
        score = unique_moves * UNIQUE_MOVE_REWARD + repeated_moves * REPEATED_MOVE_REWARD + remaining_moves * BONUS_MOVE_REWARD

    # encurage finding treasures with a multipiler
    return score * multiplier

def new_valid_agent():
    values = list(directions.values())
    rows, columns = MAZE.shape
    agent = []
    times_visited = {}
    scoring = lambda x, direction: (not x[1], times_visited.get(tuple(x[0]), 0), direction) # returns tuple (is_invalid, times_visited)
    curr = MAZE_START
    for _ in range(rows  * columns):
        moves = sorted([scoring(move(m, curr), m) for m in values])
        agent.append(moves[0][2])

        curr, _ = move(agent[-1], curr)
        curr_tuple = tuple(curr)
        times_visited[curr_tuple] = times_visited.get(curr_tuple, 0) + 1
    return agent

def generate_valid_population(n):
    # generates valid population of size 'n'
    pop = []
    for _ in range(n):
        pop.append(new_valid_agent())
    return pop


def show_solution(path, solution_fitness, sequential_display=False):
    print("-----------------------------------")
    print("Current score {}".format(solution_fitness))

    # cordinates on the path
    visited = []
    # current position
    curr = MAZE_START

    # output path commands & save visited cordinates
    for p in path:
        print(directions_reverse[p], end="")
        print(" ", end="")
        # check for wall collisions and other invalid moves
        curr, is_valid = move(p, curr)
        # the move was valid -> update current position
        if is_valid:
            visited.append((curr[0], curr[1]))
            
            # finish reached
            if MAZE[curr[1], curr[0]] == encoding["E"]:
                break
    print()

    # display maze with final solution
    display_maze(set(visited))


    # display maze move by move
    if sequential_display:
        print("--> sequential display:")
        sequential_visited = set()
        sequential_visited.add(tuple(MAZE_START))
        for v in visited:
            sequential_visited.add(v)
            display_maze(sequential_visited, current=v)
            sleep(2)
    print()

def display_maze(visited, current=None):
    # display maze with visited points marked with 'x'
    rows, columns = MAZE.shape
    for i in range(rows):
        for j in range(columns):
            if current == (j, i):
                print("o", end="")
            elif (j, i) in visited:
                print("x", end="")
            else:
                print(encoding_reverse[MAZE[i, j]], end="")
        print()

def instance_mutation(offspring, ga_instance):
    # mutate genes
    genes_to_mutate = [random() <= ga_instance.mutation_probability for _ in range(len(offspring))]

    curr = MAZE_START
    for i in range(len(offspring)):
        valid_moves = [m for m in directions.values() if move(m, curr)[1]]
        # mutate direction gene
        if genes_to_mutate[i]:
            offspring[i] = choice(valid_moves)

        curr, is_valid = move(offspring[i], curr)
        # correct invalid moves (move might be invalid because of previous mutations)
        if not is_valid:
            offspring[i] = choice(valid_moves)
            curr, _ = move(offspring[i], curr)

    return offspring


def mutation(offspring, ga_instance):
    new_offspring = []
    for o in offspring:
        new_offspring.append(instance_mutation(o, ga_instance))
    return new_offspring

def map_visited_to_indicies(path):
    # returns all visited cordinates with indicies of when they were visited and last position of the path
    curr = MAZE_START
    curr_tuple = tuple(curr)
    visited_on_indicies = {
        curr_tuple: [0]
    }
    cordinates = set()
    cordinates.add(curr_tuple)
    for i in range(len(path)):
        curr, _ = move(path[i], curr)
        curr_tuple = tuple(curr)
        visited_on_indicies.setdefault(curr_tuple, [])
        visited_on_indicies[curr_tuple].append(i + 1)
        cordinates.add(curr_tuple)
    return visited_on_indicies, cordinates, curr

def instance_crossover(A, B):
    A_visited, A_cordinates, _ = map_visited_to_indicies(A)
    B_visited, B_cordinates, B_last = map_visited_to_indicies(B)

    # find cordinates that both A and B visited (this will be at least the starting cordinate so the intersection is never empty)
    intersections = A_cordinates.intersection(B_cordinates)
    selected_inter = choice(list(intersections))

    # choose random index in both paths
    A_index = choice(A_visited[selected_inter])
    B_index = choice(B_visited[selected_inter])

    # join A and B at the selected index then make sure that there aren't too many genes
    offspring = np.concatenate((A[:A_index], B[B_index:]))[:len(A)]
    
    # if the offspring has too little genes add some random valid ones
    curr = B_last
    for _ in range(len(A) - len(offspring)):
        valid_moves = [m for m in directions.values() if move(m, curr)[1]]
        offspring = np.append(offspring, choice(valid_moves))
        curr, _ = move(offspring[-1], curr)

    return offspring

def crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(shape=offspring_size)

    for i in range(offspring_size[0]):
        child = instance_crossover(choice(parents), choice(parents))
        offspring[i] = child
    return offspring

def on_generation(ga_instance):
    if ga_instance.generations_completed % SHOW_EVERY_N_GENS == 0:
        solution, solution_fitness, _ = ga_instance.best_solution()
        show_solution(solution, solution_fitness)

if __name__ == "__main__":
    maze_file = "./mazes/maze_treasure_2.txt"
    if len(argv) > 1:
        maze_file = argv[1]

    # initalize random numbers generator
    seed(RANDOM_SEED)
    # read variables MAZE and MAZE_START from file
    encode_maze(open(maze_file, "r").read())

    # initialize population
    initial_population = np.array(generate_valid_population(80))

    # setup ga algorithem
    ga = pygad.GA(
        # main settings
        random_seed=RANDOM_SEED,
        num_generations=500,
        num_parents_mating=8,
        parent_selection_type="sus",

        # initial population
        initial_population=initial_population,

        # agent gene type
        gene_type=int,
        init_range_low=0, # lowest valid value for gene (inclusive)
        init_range_high=4, # highest valid value for gene (exclusive)

        # agent evaluation
        mutation_probability=0.1,
        keep_elitism=1, # keep best n solutions in the next generation

        # custom functions
        fitness_func=fitness,
        mutation_type=mutation,
        crossover_type=crossover,
        on_generation=on_generation,

        # computation
        parallel_processing=['thread', 16] 
    )

    # run multiple tournaments and generations to find the best solution
    ga.run()

    # read & display best solution
    solution, solution_fitness, _ = ga.best_solution()
    show_solution(solution, solution_fitness, sequential_display=True)
    
