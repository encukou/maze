#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy
cimport numpy
cimport cython


cdef extern from "stdlib.h":
    int rand()
    void srand(long int seedval)


cdef extern from "time.h":
    long int time(int)


cdef int randint(int limit):
    return rand() % limit


cpdef numpy.ndarray[numpy.int8_t, ndim=2] maze(int height, int width, double complexity=.75, double density=.75):
    """
    https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """
    # Get real random numbers
    srand(time(0))
    # Only odd shapes
    cdef numpy.ndarray[numpy.int_t, ndim=1] shape = numpy.empty(2, dtype=numpy.int)
    shape[0] = (height // 2) * 2 + 1
    shape[1] = (width // 2) * 2 + 1
    # Adjust complexity and density relative to maze size
    cdef int icomplexity = int(complexity * (5 * (shape[0] + shape[1])))
    cdef int idensity = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    cdef numpy.ndarray[numpy.int8_t, ndim=2] Zarr = numpy.zeros(shape, dtype=numpy.int8)
    cdef numpy.int8_t[:, :] Z = Zarr
    # Fill borders
    Z[0, :] = -1
    Z[shape[0] - 1, :] = -1
    Z[:, 0] = -1
    Z[:, shape[1] - 1] = -1
    # Make aisles
    cdef int i, j, x, y, x_, y_, idx
    cdef numpy.ndarray[numpy.int_t, ndim=2] nbours = numpy.empty((4, 2), dtype=numpy.int)
    for i in range(idensity):
        x, y = randint(shape[1] // 2 + 1) * 2, randint(shape[0] // 2 + 1) * 2
        Z[y, x] = -1
        for j in range(icomplexity):
            idx = 0
            if x > 1:
                nbours[idx, 0] = y
                nbours[idx, 1] = x - 2
                idx += 1
            if x < shape[1] - 2:
                nbours[idx, 0] = y
                nbours[idx, 1] = x + 2
                idx += 1
            if y > 1:
                nbours[idx, 0] = y - 2
                nbours[idx, 1] = x
                idx += 1
            if y < shape[0] - 2:
                nbours[idx, 0] = y + 2
                nbours[idx, 1] = x
                idx += 1
            if idx:
                y_, x_ = nbours[randint(idx), 0], nbours[randint(idx), 1]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = -1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = -1
                    x, y = x_, y_
    # Add random castle
    x = randint(shape[1] - 2) + 1
    y = randint(shape[0] - 2) + 1
    Z[y, x] = 1
    # Add one random dude
    x_, y_ = x, y
    while x_ == x and y_ == y:
        x_ = randint(shape[1] - 2) + 1
        y_ = randint(shape[0] - 2) + 1
    Z[y_, x_] = 2
    return Zarr
