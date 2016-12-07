#cython: language_level=3
import numpy
cimport numpy
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


cdef struct coords:
    int r
    int c


cdef coords up(coords shape, coords loc):
    if loc.r == 0:
        return coords(-1, -1)
    return coords(loc.r - 1, loc.c)


cdef coords down(coords shape, coords loc):
    if loc.r == shape.r - 1:
        return coords(-1, -1)
    return coords(loc.r + 1, loc.c)


cdef coords left(coords shape, coords loc):
    if loc.c == 0:
        return coords(-1, -1)
    return coords(loc.r, loc.c - 1)


cdef coords right(coords shape, coords loc):
    if loc.c == shape.c - 1:
        return coords(-1, -1)
    return coords(loc.r, loc.c + 1)


def ends(maze):
    return numpy.asarray(numpy.where(maze == 1)).T


def starts(maze):
    return numpy.asarray(numpy.where(maze >= 2)).T


cdef char TARGET = ord('X')
cdef char LEFT = ord('<')
cdef char RIGHT = ord('>')
cdef char UP = ord('^')
cdef char DOWN = ord('v')
cdef char WALL = ord('#')
cdef char SPACE = ord(' ')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def arrows_to_path(numpy.ndarray[numpy.int8_t, ndim=2] arrows, int column, int row):
    cdef coords loc = coords(column, row)
    cdef coords shape
    shape.r, shape.c = arrows.shape[0], arrows.shape[1]

    if arrows[loc.r, loc.c] == WALL:
        raise ValueError('Cannot construct path for wall')
    if arrows[loc.r, loc.c] == SPACE:
        raise ValueError('Cannot construct path for unreachable cell')

    # the path can never be longer than number of cells
    cdef coords * path = <coords *>PyMem_Malloc(shape.r*shape.c*sizeof(coords))
    if path == NULL:
        raise MemoryError()
    path[0] = loc
    cdef size_t s = 1
    cdef char symb
    cdef coords nloc = loc
    while arrows[nloc.r, nloc.c] != TARGET:
        symb = arrows[nloc.r, nloc.c]
        if symb == UP:
            nloc = up(shape, nloc)
        elif symb == LEFT:
            nloc = left(shape, nloc)
        elif symb == RIGHT:
            nloc = right(shape, nloc)
        else:
            nloc = down(shape, nloc)
        path[s] = nloc
        s += 1

    lpath = []
    cdef size_t i
    for i in range(s):
        lpath.append((path[i].r, path[i].c))

    PyMem_Free(path)
    return lpath


cdef struct job:
    coords loc
    int dist
    char symb


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef class JobQueue:
    cdef job * jobs
    cdef int top, bottom, size

    def __cinit__(self, int size):
        self.jobs = <job *>PyMem_Malloc(size*sizeof(job))
        if self.jobs == NULL:
            raise MemoryError()
        self.top = 0
        self.bottom = 0
        self.size = size

    def __dealloc__(self):
        if self.jobs != NULL:
            PyMem_Free(self.jobs)

    cdef void put(self, job ajob):
        self.jobs[self.top % self.size] = ajob
        self.top += 1

    cdef job get(self):
        self.bottom += 1
        return self.jobs[(self.bottom-1) % self.size]

    cdef bint empty(self):
        return self.bottom == self.top


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def flood(numpy.ndarray[numpy.int8_t, ndim=2] maze):
    cdef coords shape = coords(maze.shape[0], maze.shape[1])
    cdef numpy.ndarray[numpy.int_t, ndim=2] distances = numpy.full((shape.r, shape.c), -1, dtype=numpy.int)

    # int8_t is the same type as ('a', 1), but we'll need to work with ords
    # Initialize everything as walls
    # Cannot use WALL here, Python is handling this, not C
    cdef numpy.ndarray[numpy.int8_t, ndim=2]  directions = numpy.full((shape.r, shape.c), b'#', dtype=('a', 1))
    # Add spaces where there are no walls
    # Cannot use SPACE here, dtto
    directions[maze >= 0] = b' '

    # TODO allocate what we actually need, this is a guess
    ends_ = ends(maze)
    cdef JobQueue jobs = JobQueue(shape.r*shape.c*min(len(ends_), 4))

    for end in ends_:
        jobs.put(job(coords(end[0], end[1]), 0, TARGET))

    cdef coords loc, nloc
    cdef size_t i
    cdef int dist
    cdef char symb
    cdef job ajob
    while not jobs.empty():
        ajob = jobs.get()
        loc = ajob.loc
        dist = ajob.dist
        symb = ajob.symb
        # It's a wall or we've been there better
        if directions[loc.r, loc.c] == WALL or 0 <= distances[loc.r, loc.c] <= dist:
            continue
        directions[loc.r, loc.c] = symb
        distances[loc.r, loc.c] = dist

        nloc = down(shape, loc)
        if nloc.r != -1:
            jobs.put(job(nloc, dist+1, UP))

        nloc = up(shape, loc)
        if nloc.r != -1:
            jobs.put(job(nloc, dist+1, DOWN))

        nloc = left(shape, loc)
        if nloc.r != -1:
            jobs.put(job(nloc, dist+1, RIGHT))

        nloc = right(shape, loc)
        if nloc.r != -1:
            jobs.put(job(nloc, dist+1, LEFT))

    return distances, directions


def create_lines(arrows, locations):
    ret = []
    for loc in locations:
        loc = tuple(loc)
        try:
            ret.append(arrows_to_path(arrows, *loc))
        except ValueError:
            # unreachable
            pass
    return ret


def is_reachable(arrows):
    return b' ' not in arrows


class AnalyzedMaze:
    def __init__(self, maze):
        self.distances, self.directions = flood(maze)
        self.lines = create_lines(self.directions, starts(maze))
        self.is_reachable = is_reachable(self.directions)

    def path(self, column, row):
        return arrows_to_path(self.directions, column, row)


def analyze(maze):
    return AnalyzedMaze(maze)
