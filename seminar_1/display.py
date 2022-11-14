from time import sleep


def show_solution(maze_ga, path, solution_fitness, sequential_display=False):
    print("-----------------------------------")
    print("Current score {}".format(solution_fitness))

    # cordinates on the path
    visited = []
    # current position
    curr = maze_ga.maze_start

    # output path commands & save visited cordinates
    for p in path:
        print(maze_ga.directions_reverse[p], end="")
        print(" ", end="")
        # check for wall collisions and other invalid moves
        curr, is_valid = maze_ga.move(p, curr)
        # the move was valid -> update current position
        if is_valid:
            visited.append((curr[0], curr[1]))
            
            # finish reached
            if maze_ga.maze[curr[1], curr[0]] == maze_ga.encoding["E"]:
                break
    print()

    # display maze with final solution
    display_maze(maze_ga, set(visited))


    # display maze move by move
    if sequential_display:
        print("--> sequential display:")
        sequential_visited = set()
        sequential_visited.add(tuple(maze_ga.maze_start))
        for v in visited:
            sequential_visited.add(v)
            display_maze(maze_ga, sequential_visited, current=v)
            sleep(2)
    print()

def display_maze(maze_ga, visited, current=None):
    # display maze with visited points marked with 'x'
    rows, columns = maze_ga.maze.shape
    for i in range(rows):
        for j in range(columns):
            if current == (j, i):
                print("o", end="")
            elif (j, i) in visited:
                print("x", end="")
            else:
                print(maze_ga.encoding_reverse[maze_ga.maze[i, j]], end="")
        print()

def on_generation_factory(maze_ga, each_n_generations=1, wait_on_show=False):
    def on_generation(ga_instance):
        print("generation: {}".format(ga_instance.generations_completed))
        if ga_instance.generations_completed % each_n_generations == 0:
            solution, solution_fitness, _ = ga_instance.best_solution()
            show_solution(maze_ga, solution, solution_fitness)
            if wait_on_show:
                input("--> Press Any Key to continue...")
    return on_generation