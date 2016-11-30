import queue

import numpy


def up(maze, loc):
    if loc[0] == 0:
        raise ValueError
    return loc[0] - 1, loc[1]


def down(maze, loc):
    if loc[0] == (maze.shape[0] - 1):
        raise ValueError
    return loc[0] + 1, loc[1]


def left(maze, loc):
    if loc[1] == 0:
        raise ValueError
    return loc[0], loc[1] - 1


def right(maze, loc):
    if loc[1] == (maze.shape[1] - 1):
        raise ValueError
    return loc[0], loc[1] + 1


def ends(maze):
    return numpy.asarray(numpy.where(maze == 1)).T


DIRS = {
    b'^': up,
    b'<': left,
    b'>': right,
    b'v': down,
}

ANTIDIRS = {
    down: b'^',
    right: b'<',
    left: b'>',
    up: b'v'
}


def arrows_to_path(arrows, loc):
    if arrows[loc] == b'#':
        raise ValueError('Cannot construct path for wall')
    if arrows[loc] == b' ':
        raise ValueError('Cannot construct path for unreachable cell')
    path = [loc]

    nloc = loc
    while arrows[nloc] != b'X':
        nloc = DIRS[arrows[nloc]](arrows, nloc)
        path.append(nloc)

    return path


def flood(maze):
    distances = numpy.full(maze.shape, -1, dtype=numpy.int)

    # Initialize everything as walls
    directions = numpy.full(maze.shape, b'#', dtype=('a', 1))
    # Add spaces where there are no walls
    directions[maze >= 0] = b' '

    jobs = queue.Queue()
    for end in ends(maze):
        jobs.put((tuple(end), 0, b'X'))

    while not jobs.empty():
        loc, dist, char = jobs.get()
        # It's a wall or we've been there better
        if directions[loc] == b'#' or 0 <= distances[loc] <= dist:
            continue
        directions[loc] = char
        distances[loc] = dist
        for func in [up, left, right, down]:
            try:
                jobs.put((func(maze, loc), dist+1, ANTIDIRS[func]))
            except ValueError:
                # Out of matrix
                pass

    return distances, directions


def is_reachable(arrows):
    return b' ' not in arrows


class AnalyzedMaze:
    def __init__(self, maze):
        self.distances, self.directions = flood(maze)
        self.is_reachable = is_reachable(self.directions)

    def path(self, column, row):
        return arrows_to_path(self.directions, (column, row))


def analyze(maze):
    return AnalyzedMaze(maze)
