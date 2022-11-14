from random import choice, choices, random

import numpy as np


def fitness_factory(maze_ga):
    def fitness(path, solution_idx):
        curr = maze_ga.maze_start
        visited = set()
        visited.add(tuple(curr))

        collected_treasures = 0
        multiplier = 1
        moves = 0
        invalid_moves = 0
        found_finish = False
        for i in range(len(path)):
            moves += 1
            p = path[i]

            # check if the move is valid and if any score adjustment is required
            curr, is_valid = maze_ga.move(p, curr)
            curr_tuple = tuple(curr)

            if not is_valid:
                invalid_moves += 1
            
            # algorithem found treasure
            if curr_tuple not in visited and maze_ga.maze[curr[1], curr[0]] == maze_ga.encoding["T"]:
                collected_treasures += 1
                multiplier += maze_ga.TREASURE_MULTIPLIER

            # algorithem found finish
            if maze_ga.maze[curr[1], curr[0]] == maze_ga.encoding["E"]:
                found_finish = True
                multiplier += maze_ga.FINISH_MULTPILER
                break
            
            visited.add(curr_tuple)

        # define move counters
        unique_moves = len(visited)
        repeated_moves = moves - unique_moves
        remaining_moves = len(path) - moves    

        # encurage exploration
        score = unique_moves * maze_ga.UNIQUE_MOVE_REWARD

        # if the maze was solved with all treasures collected
        if found_finish and collected_treasures == maze_ga.treasures:
            # we encurage making unique moves, but don't punish repeated ones too much, we also highly encurage using as little moves as possible using bonus reward
            score = unique_moves * maze_ga.UNIQUE_MOVE_REWARD + repeated_moves * maze_ga.REPEATED_MOVE_REWARD + remaining_moves * maze_ga.BONUS_MOVE_REWARD

        # encurage finding treasures with a multipiler
        return score * multiplier - invalid_moves * maze_ga.INVALID_MOVE_PENALTY
    return fitness

def instance_mutation_factory(maze_ga):
    def instance_mutation(offspring, ga_instance):
        # mutate genes
        genes_to_mutate = [random() <= ga_instance.mutation_probability for _ in range(len(offspring))]

        curr = maze_ga.maze_start
        for i in range(len(offspring)):
            valid_moves = [m for m in maze_ga.directions.values() if maze_ga.move(m, curr)[1]]
            # mutate direction gene
            if genes_to_mutate[i]:
                offspring[i] = choice(valid_moves)

            curr, is_valid = maze_ga.move(offspring[i], curr)
            # correct invalid moves (move might be invalid because of previous mutations)
            if not is_valid:
                offspring[i] = choice(valid_moves)
                curr, _ = maze_ga.move(offspring[i], curr)

        return offspring
    return instance_mutation


def mutation_factory(maze_ga):
    instance_mutation = instance_mutation_factory(maze_ga)
    def mutation(offspring, ga_instance):
        new_offspring = []
        for o in offspring:
            new_offspring.append(instance_mutation(o, ga_instance))
        return new_offspring
    return mutation

def map_visited_to_indicies(maze_ga, path):
        # returns all visited cordinates with indicies of when they were visited and last position of the path
        curr = maze_ga.maze_start
        curr_tuple = tuple(curr)
        visited_on_indicies = {
            curr_tuple: [0]
        }
        cordinates = set()
        cordinates.add(curr_tuple)
        for i in range(len(path)):
            curr, _ = maze_ga.move(path[i], curr)
            curr_tuple = tuple(curr)
            visited_on_indicies.setdefault(curr_tuple, [])
            visited_on_indicies[curr_tuple].append(i + 1)
            cordinates.add(curr_tuple)
        return visited_on_indicies, cordinates, curr

def instance_crossover_factory(maze_ga):
    def crossover(A, B):
        A_visited, A_cordinates, _ = map_visited_to_indicies(maze_ga, A)
        B_visited, B_cordinates, B_last = map_visited_to_indicies(maze_ga, B)

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
            valid_moves = [m for m in maze_ga.directions.values() if maze_ga.move(m, curr)[1]]
            offspring = np.append(offspring, choice(valid_moves))
            curr, _ = maze_ga.move(offspring[-1], curr)

        return offspring
    return crossover

def crossover_factory(maze_ga):
    instance_crossover = instance_crossover_factory(maze_ga)
    def crossover(parents, offspring_size, ga_instance):
        offspring = np.empty(shape=offspring_size)

        for i in range(offspring_size[0]):
            child = instance_crossover(choice(parents), choice(parents))
            offspring[i] = child
        return offspring
    return crossover


def valid_agent_factory(maze_ga):
    def new_agent():
        values = list(maze_ga.directions.values())
        rows, columns = maze_ga.maze.shape
        agent = []
        times_visited = {}
        scoring = lambda x, direction: (not x[1], times_visited.get(tuple(x[0]), 0), direction) # returns tuple (is_invalid, times_visited)
        curr = maze_ga.maze_start
        for _ in range(rows  * columns):
            moves = sorted([scoring(maze_ga.move(m, curr), m) for m in values])
            agent.append(moves[0][2])

            curr, _ = maze_ga.move(agent[-1], curr)
            curr_tuple = tuple(curr)
            times_visited[curr_tuple] = times_visited.get(curr_tuple, 0) + 1
        return agent
    return new_agent


def agent_factory(maze_ga):
    values = list(maze_ga.directions.values())
    def new_agent():
        return choices(values, k=maze_ga.maze.size)
    return new_agent


def generate_population_factory(maze_ga, valid_only=False):
    generator = agent_factory(maze_ga)
    if valid_only:
        generator = valid_agent_factory(maze_ga)
    
    def generate_population(population_size):
        return np.array([generator() for _ in range(population_size)])
    return generate_population