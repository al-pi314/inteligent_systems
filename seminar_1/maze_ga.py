import numpy as np

encoding = {
    "#": 0,
    ".": 1,
    "S": 2,
    "E": 3,
    "T": 4,
    "x": 5,
}

directions = {
    bin(0): "S",
    bin(1): "L",
    bin(2): "U",
    bin(3): "R",
    bin(4): "D"
}

def encode_maze(s):
    rows = s.split("\n")
    arr = np.zeros((len(rows), len(rows[0])))

    start = None
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            val = rows[i][j]
            arr[i, j] = encoding[val]
            if val == "S":
                start = [j, i]

    return arr, start

def interpret(move):
    return directions[move]

def fitness(maze, maze_start, logic):
    score = -5
    moves_cnt = 0

    rows, columns = maze.shape
    curr = maze_start
    finish_reached = False
    for i in range(maze.size):
        c = interpret(next_move(maze, logic))
        if c == "S":
            break
        n_curr = curr[:]
        moves_cnt += 1

        # move
        if c == "L":
            n_curr[0] -= 1
        elif c == "R":
            n_curr[0] += 1
        elif c == "U":
            n_curr[1] -=1
        elif c == "D":
            n_curr[1] += 1

        # checks
        # out of bounds
        if n_curr[0] < 0 or n_curr[0] >= columns or n_curr[1] < 0 or n_curr[1]  >= rows:
            score -= 1
        # in wall
        elif maze[n_curr[1], n_curr[0]] == encoding["#"]:
            score -= 1
        else:
            maze[curr[1], curr[0]] = encoding["x"]
            curr = n_curr

        # algorithem found treasure
        if maze[curr[1], curr[0]] == encoding["T"]:
            score += 10
            maze[curr[1], curr[0]] = encoding["."]

        # algorithem found finish
        if not finish_reached and maze[curr[1], curr[0]] == encoding["E"]:
            finish_reached = True
            score += 100
        
        maze[curr[1], curr[0]] = encoding["S"]

    # penalise making more steps
    return score / max(1, moves_cnt)

def next_move(maze, logic):


if __name__ == "__main__":
    print(encode_maze(open("./mazes/maze_example.txt", "r").read()))
