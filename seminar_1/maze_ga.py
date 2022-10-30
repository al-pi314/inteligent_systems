

from random import choice, seed, random
from sys import argv
import numpy as np
import pygad

# global variables
RANDOM_SEED = 100
MAZE = None
MAZE_START = None
SCORING_FACTOR = 1

# factor to use for penalties & rewards
FINISHING_MOVE = 1
TREASURE_MOVE = 1
INVALID_MOVE = 0.01
STARTING_SCORE = -0.1

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
    global MAZE, MAZE_START, SCORING_FACTOR

    rows = s.split("\n")
    MAZE = np.zeros((len(rows), len(rows[0])))
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            val = rows[i][j]
            MAZE[i, j] = encoding[val]
            if val == "S":
                MAZE_START = [j, i]

    SCORING_FACTOR = (MAZE.size / 100)


def move(p, n_curr):
    rows, columns = MAZE.shape
    
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
        return False
    # in wall
    elif MAZE[n_curr[1], n_curr[0]] == encoding["#"]:
        return False
    # valid move
    return True


def fitness(path, solution_idx):
    moves_cnt = 0
    # set of collected threasure points to prevent collecting the same threasure twice
    collected_threasures = set()

    curr = MAZE_START
    score = STARTING_SCORE * SCORING_FACTOR
    for p in path:
        moves_cnt += 1
        
        # check if the move is valid and if any score adjustment is required
        n_curr = curr[:]
        is_valid = move(p, n_curr)
        if is_valid:
            # move was valid -> update current position
            curr = n_curr
        else:
            # move was invalid -> adjust score
            score -= INVALID_MOVE * SCORING_FACTOR

        # algorithem found treasure
        if (curr[0], curr[1]) not in collected_threasures and MAZE[curr[1], curr[0]] == encoding["T"]:
            score += TREASURE_MOVE * SCORING_FACTOR
            collected_threasures.add((curr[0], curr[1]))

        # algorithem found finish
        if MAZE[curr[1], curr[0]] == encoding["E"]:
            score += FINISHING_MOVE * SCORING_FACTOR
            # stop after encountering the finish
            break
        
    # penalise making more steps
    return score / max(1, moves_cnt)

def new_agent():
    # generate new agent (genes are selected with probability distribution)
    values = list(directions.values())
    rows, columns = MAZE.shape
    return [choice(values) for _ in range(rows  * columns)]

def generate_population(n):
    # generate population of size 'n'
    pop = []
    for _ in range(n):
        pop.append(new_agent())
    return pop

def show_solution(path):
    # cordinates on the path
    visited = set()
    # current position
    curr = MAZE_START

    # output path commands & save visited cordinates
    for p in path:
        print(directions_reverse[p], end="")
        print(" ", end="")
        # check for wall collisions and other invalid moves
        n_curr = curr[:]
        is_valid = move(p, n_curr)
        # the move was valid -> update current position
        if is_valid:
            curr = n_curr
            visited.add((curr[0], curr[1]))
            
            # finish reached
            if MAZE[curr[1], curr[0]] == encoding["E"]:
                break
    print()

    # display maze with visited points marked with 'x'
    rows, columns = MAZE.shape
    for i in range(rows):
        for j in range(columns):
            if (j, i) in visited:
                print("x", end="")
            else:
                print(encoding_reverse[MAZE[i, j]], end="")
        print()
    print()

def instance_mutation(offspring, ga_instance):
    genes_to_mutate = [random() <= ga_instance.mutation_probability for _ in range(len(offspring))]

    curr = MAZE_START
    for i in range(len(offspring)):
        n_curr = curr[:]
        if genes_to_mutate[i]:
            valid_moves = [m for m in directions.values() if move(m, n_curr)]
            offspring[i] = choice(valid_moves)

        is_valid = move(offspring[i], n_curr)
        if is_valid:
            curr = n_curr
    return offspring

def mutation(offspring, ga_instance):
    new_offspring = []
    for o in offspring:
        new_offspring.append(instance_mutation(o, ga_instance))
    return new_offspring

def instance_crossover(A, B):
    suboptimal_moves = []
    curr = MAZE_START
    for i in range(len(A)):
        n_curr = curr[:]
        is_valid = move(A[i], n_curr)
        if is_valid:
            curr = n_curr
        else:
            suboptimal_moves.append(i)
    if len(suboptimal_moves) == 0:
        return A

    swap_idx = choice(suboptimal_moves)
    return np.concatenate((A[:swap_idx], B[swap_idx:]))

def crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(shape=offspring_size)

    for i in range(offspring_size[0]):
        child = instance_crossover(choice(parents), choice(parents))
        offspring[i] = child
    return offspring

if __name__ == "__main__":
    maze_file = "./mazes/maze_treasure_2.txt"
    if len(argv) > 1:
        maze_file = argv[1]

    # initalize random numbers generator
    seed(RANDOM_SEED)
    # read variables MAZE and MAZE_START from file
    encode_maze(open(maze_file, "r").read())

    # initialize population
    initial_population = np.array(generate_population(300))

    # setup ga algorithem
    ga = pygad.GA(
        # main settings
        random_seed=RANDOM_SEED,
        num_generations=150,
        num_parents_mating=30,
        K_tournament=10,

        # initial population
        initial_population=initial_population,

        # agent gene type
        gene_type=int,
        init_range_low=0, # lowest valid value for gene (inclusive)
        init_range_high=4, # highest valid value for gene (exclusive)

        # agent evaluation
        mutation_probability=0.3,
        keep_elitism=10, # keep best n solutions in the next generation

        # custom functions
        fitness_func=fitness,
        mutation_type=mutation,
        crossover_type=crossover,

        # computation
        parallel_processing=['thread', 5] 
    )

    # run multiple tournaments and generations to find the best solution
    ga.run()

    # read & display best solution
    solution, solution_fitness, solution_idx = ga.best_solution()
    show_solution(solution)
    
