import numpy
from numpy.random import choice

UP = 2**0
LEFT = 2**1
DOWN = 2**2
RIGHT = 2**3


def add_line(sequence, *, seen, array=None, shape=None):
    """
    Constructs an array with a bit-encoded line.
    On each cell, there might be 4 line segments:

    - up (bit 0)
    - left (bit 1)
    - down (bit 2)
    - right (bit 3)

    By the combination of those, any path can be constructed.

    Args:
        sequence: An iterable with tuple-like coordinates, in order of line
        array: Numpy array to modify (new is created if omitted)
        shape: If no array is provided, the shape of the new one

    Returns:
        ndarray: Modified or created array
    """
    if array is None:
        if shape is None:
            raise ValueError('Either array or shape argument is mandatory')
        array = numpy.zeros(shape, dtype=numpy.uint8)

    last = None
    for cell in sequence:
        if not last:
            last = cell
            continue
        d, o = _direction(last, cell)
        array[tuple(last)] |= d
        array[tuple(cell)] |= o
        if cell in seen:
            # already on a path that exists
            # need to do this after processing the cell
            # because first cell of a common path is the same
            # but may come form a different direction
            # we can also only do this optimization if paths are deterministic
            break
        seen.add(cell)
        last = cell

    return array


def add_lines(sequences, *, array=None, shape=None):
    seen = set()
    array = add_line([], array=array, shape=shape, seen=seen)
    for sequence in sequences:
        add_line(sequence, array=array, seen=seen)
    return array


def _direction(first, second):
    diff = (second[0] - first[0], second[1] - first[1])
    if diff == (0, 1):
        return RIGHT, LEFT
    if diff == (0, -1):
        return LEFT, RIGHT
    if diff == (1, 0):
        return DOWN, UP
    if diff == (-1, 0):
        return UP, DOWN
    raise ValueError('The two cells provided are not next to each other')


def _randompath(start=[0, 0], length=20):
    """
    Get a random path.
    Just for debugging, it will run out of the shape if long enough
    """
    sequence = [start]
    for i in range(length - 1):
        next = [sequence[i][0], sequence[i][1]]
        next[choice((0, 1))] += 1
        sequence.append(next)
    return sequence
