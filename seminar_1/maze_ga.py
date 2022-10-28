

from random import choices, seed

import numpy as np
import pygad

# global variables
RANDOM_SEED = 100
MAZE = None
MAZE_START = None

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
    "S": 0,
    "L": 1,
    "R": 2,
    "U": 3,
    "D": 4
}
directions_reverse = {v: k for k, v in directions.items()}

def encode_maze(s):
    # sets global variables of MAZE and MAZE_START based on encoding values
    global MAZE, MAZE_START

    rows = s.split("\n")
    MAZE = np.zeros((len(rows), len(rows[0])))
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            val = rows[i][j]
            MAZE[i, j] = encoding[val]
            if val == "S":
                MAZE_START = [j, i]


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
        return -1
    # in wall
    elif MAZE[n_curr[1], n_curr[0]] == encoding["#"]:
        return -1
    # valid move
    return 0


def fitness(path, solution_idx):
    score = -5
    moves_cnt = 0
    # set of collected threasure points to prevent collecting the same threasure twice
    collected_threasures = set()

    curr = MAZE_START
    finish_reached = False
    for p in path:
        moves_cnt += 1
        # agent decided to stop
        if p == directions["S"]:
            break
        
        # check if the move is valid and if any score adjustment is required
        n_curr = curr[:]
        score_adjustment = move(p, n_curr)
        if score_adjustment == 0:
            # move was valid -> update current position
            curr = n_curr
        else:
            # move was invalid -> adjust score
            score += score_adjustment

        # algorithem found treasure
        if (curr[0], curr[1]) not in collected_threasures and MAZE[curr[1], curr[0]] == encoding["T"]:
            score += 10
            collected_threasures.add((curr[0], curr[1]))

        # algorithem found finish
        if not finish_reached and MAZE[curr[1], curr[0]] == encoding["E"]:
            finish_reached = True
            score += 100
        
    # penalise making more steps
    return score / max(1, moves_cnt)

def new_agent():
    # generate new agent (genes are selected with probability distribution)
    active_moves = sorted(list(directions.values()))
    weights = [0.03] + [0.2425 for _ in range(len(directions) -1)]
    return choices(active_moves, weights, k=MAZE.size)

def generate_population(n):
    # generate population of size 'n'
    pop = []
    for _ in range(n):
        na = new_agent()
        pop.append(na)
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
        sa = move(p, n_curr)
        # the move was not invalid -> update current position
        if sa == 0:
            curr = n_curr
            visited.add((n_curr[0], n_curr[1]))
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

if __name__ == "__main__":
    # initalize random numbers generator
    seed(RANDOM_SEED)
    # read variables MAZE and MAZE_START from file
    encode_maze(open("./mazes/maze_treasure_2.txt", "r").read())

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

        # agent evaluation
        mutation_probability=0.4,
        fitness_func=fitness,
        keep_elitism=5, # keep best n solutions in the next generation

        # agent gene type
        gene_type=int,
        init_range_low=0, # lowest valid value for gene (inclusive)
        init_range_high=5, # highest valid value for gene (exclusive)

        # computation
        parallel_processing=['thread', 4] 
    )

    # run multiple tournaments and generations to find the best solution
    ga.run()

    # read & display best solution
    solution, solution_fitness, solution_idx = ga.best_solution()
    show_solution(solution)
    
