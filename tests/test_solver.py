from itertools import product

import numpy
import pytest

from maze import analyze


S = (1, 5, 20, 100, 200)
VERTICAL = 'vertical'
HORIZONTAL = 'horizontal'
D = (VERTICAL, HORIZONTAL)


def skip_small(h, w, d):
    if d == HORIZONTAL and h < 5 or d == VERTICAL and w < 5:
        pytest.skip('to small to test')


def skip_large(h, w):
    if h > 100 and w > 100:
        pytest.skip('to large to test')


def ids(things):
    return '-'.join(map(str, things))


@pytest.fixture(scope='module', params=product(S, S), ids=ids)
def global_empty(request):
    h, w = request.param
    maze = zeros(h, w)
    maze[0, 0] = 1
    amaze = analyze(maze)
    return maze, amaze


@pytest.fixture
def empty(global_empty):
    print_all(*global_empty[:2])
    return global_empty


def test_empty_shape(empty):
    maze, amaze = empty
    check_shape(maze, amaze)


def test_empty_reachable(empty):
    maze, amaze = empty
    assert amaze.is_reachable


def test_empty_only_top_left_X(empty):
    maze, amaze = empty
    check_only(amaze.directions, (b'^<X'))


def test_empty_X00(empty):
    maze, amaze = empty
    check_X00(amaze)


def test_empty_meshgrid(empty):
    maze, amaze = empty
    check_meshgrid(amaze.distances)


def test_empty_obvious_paths(empty):
    maze, amaze = empty
    check_obvious_paths(*maze.shape, amaze)


def test_empty_path_lens(empty):
    maze, amaze = empty
    check_path_lens(*maze.shape, amaze)


def test_empty_path_distance_descends(empty):
    maze, amaze = empty
    check_path_distance_descends(*maze.shape, amaze)


@pytest.fixture(scope='module', params=product(S, S, D), ids=ids)
def global_walled(request):
    h, w, d = request.param
    skip_small(h, w, d)
    maze = zeros(h, w)
    maze[0, 0] = 1
    half = w//2 if d == VERTICAL else h//2

    maze[dim(half, d)] = -1
    maze[dim(half + 1, d)] = -1
    amaze = analyze(maze)
    return maze, amaze, half, d


@pytest.fixture
def walled(global_walled):
    print_all(*global_walled[:2])
    return global_walled


def test_walled_shape(walled):
    maze, amaze, *_ = walled
    check_shape(maze, amaze)


def test_walled_not_reachable(walled):
    _, amaze, *_ = walled
    assert not amaze.is_reachable


def test_walled_only_top_left_X_space_wall(walled):
    _, amaze, *_ = walled
    check_only(amaze.directions, (b'^<X #'))


def test_walled_X00(walled):
    _, amaze, *_ = walled
    check_X00(amaze)


def test_walled_first_wall(walled):
    _, amaze, half, d = walled
    assert (amaze.directions[dim(half, d)] == b'#').all()


def test_walled_second_wall(walled):
    _, amaze, half, d = walled
    assert (amaze.directions[dim(half + 1, d)] == b'#').all()


def test_walled_behind_wall(walled):
    _, amaze, half, d = walled
    assert (amaze.directions[dim(half + 2, d)] == b' ').all()


def test_walled_only_top_left(walled):
    _, amaze, half, d = walled
    check_only(amaze.directions[dim(half - 1, d)], (b'^<'))


def test_walled_meshgrid(walled):
    _, amaze, half, d = walled
    check_meshgrid(amaze.distances[dim(slice(0, half), d)])


def test_walled_obvious_paths(walled):
    maze, amaze, half, d = walled
    check_obvious_paths(*size(*maze.shape, half, d), amaze)


def test_walled_path_lens(walled):
    maze, amaze, half, d = walled
    check_path_lens(*size(*maze.shape, half, d), amaze)


def test_walled_path_distance_descends(walled):
    maze, amaze, half, d = walled
    check_path_distance_descends(*size(*maze.shape, half, d), amaze)


def test_walled_path_raises(walled):
    maze, amaze, half, d = walled
    check_path_raises(*maze.shape, *bounds(half, d), amaze)


@pytest.fixture(scope='module', params=product(S, S), ids=ids)
def global_s_shape(request):
    h, w = request.param
    maze = numpy.full((h, w), -1, dtype=numpy.int8)
    directions = numpy.full((h, w), b'#', dtype=('a', 1))
    distances = numpy.full(maze.shape, -1, dtype=numpy.int)
    path = []

    # This prepares both the maze and expected result for shapes like:
    #
    #    v#>>v#X
    #    v#^#v#^
    #    v#^#v#^
    #    v#^#v#^
    #    v#^#v#^
    #    >>^#>>^
    for column in range(w):
        if column % 2 == 0:
            maze[:, column] = 0
            if column % 4 == 0:
                directions[:, column] = b'v'
                directions[h-1, column] = b'>'
                path += [(r, column) for r in range(h)]
            else:
                directions[:, column] = b'^'
                directions[0, column] = b'>'
                path += [(r, column) for r in reversed(range(h))]
        else:
            row = h-1 if column % 4 == 1 else 0
            path.append((row, column))
            maze[row, column] = 0
            directions[row, column] = b'>'

    # Target
    column = w-1
    row = h-1 if column % 4 < 2 else 0
    maze[row, column] = 1
    directions[row, column] = b'X'

    # Reconstruct the distances from path
    for idx, loc in enumerate(reversed(path)):
        distances[loc] = idx

    amaze = analyze(maze)
    return maze, distances, directions, path, amaze


@pytest.fixture
def s_shape(global_s_shape):
    print_all(global_s_shape[0], global_s_shape[-1])
    return global_s_shape


def test_s_shape_shape(s_shape):
    maze, *_, amaze = s_shape
    check_shape(maze, amaze)


def test_s_shape_distances(s_shape):
    maze, distances, *_, amaze = s_shape
    assert (amaze.distances == distances).all()


def test_s_shape_directions(s_shape):
    maze, _, directions, _, amaze = s_shape
    assert (amaze.directions == directions).all()


def test_s_shape_paths(s_shape):
    maze, *_, path, amaze = s_shape
    skip_large(*maze.shape)
    for loc in path:
        assert lt(amaze.path(*loc)) == path
        path = path[1:]


@pytest.fixture(scope='module')
def huge(request):
    maze = zeros(2048, 2048)
    maze[0, 0] = 1
    return maze


@pytest.mark.timeout(20)
def test_analyze_speed(huge):
    for i in range(20):
        amaze = analyze(huge)


@pytest.mark.timeout(5)
def test_path_speed(huge):
    amaze = analyze(huge)
    for i in range(250):
        amaze.path(2047, 2047)


# Helper functions bellow


def print_all(maze, amaze):
    print(maze)
    print(amaze.directions)
    print(amaze.distances)


def check_shape(maze, amaze):
    assert amaze.directions.shape == maze.shape
    assert amaze.distances.shape == maze.shape


def check_x_count(amaze, count):
    assert (amaze.directions == b'X').sum() == count


def check_meshgrid(array):
    sh = numpy.arange(array.shape[0])
    sw = numpy.arange(array.shape[1])
    mgy, mgx = numpy.meshgrid(sw, sh)
    assert (array == mgy + mgx).all()


def check_only(array, only):
    if isinstance(only, bytes):
        only = (bytes([b]) for b in only)
    bools = numpy.zeros(array.shape, dtype=numpy.bool)
    for item in only:
        bools |= (array == numpy.array([item]))
        print((array == numpy.array([item])))
    assert bools.all()


def check_X00(amaze):
    check_x_count(amaze, 1)
    assert amaze.directions[0, 0] == b'X'


def lt(thing):
    """If possible, make that thing a list of tuples"""
    return [tuple(c) for c in thing]


def check_obvious_paths(h, w, amaze):
    # There are only several locations with only one path
    # All are located in first row/column
    for row in range(h):
        assert lt(amaze.path(row, 0)) == \
            [(r, 0) for r in reversed(range(row+1))]
    for column in range(w):
        assert lt(amaze.path(0, column)) == \
            [(0, c) for c in reversed(range(column+1))]


def check_path_lens(h, w, amaze):
    # There are more possibilities, but the len is set in stone
    for row in range(1, h):
        for column in range(1, w):
            assert len(lt(amaze.path(row, column))) == row + column + 1


def check_path_distance_descends(h, w, amaze):
    # We can check that all paths leads to lower distances
    for row in range(h):
        for column in range(w):
            path = amaze.path(row, column)
            last = float('inf')  # cannot do int infinity
            for step in path:
                current = amaze.distances[tuple(step)]
                assert current < last
                last = current


def check_path_raises(h, w, r, c, amaze):
    for row in range(r, h):
        for column in range(c, w):
            with pytest.raises(BaseException):
                amaze.path(row, column)


def zeros(h, w):
    return numpy.zeros((h, w), dtype=numpy.int8)


def dim(index, direction):
    if direction == HORIZONTAL:
        return (index, ...)
    if direction == VERTICAL:
        return (..., index)
    raise ValueError('invalid direction')


def size(h, w, half, direction):
    if direction == HORIZONTAL:
        return (half, w)
    if direction == VERTICAL:
        return (h, half)
    raise ValueError('invalid direction')


def bounds(half, direction):
    if direction == HORIZONTAL:
        return (half, 0)
    if direction == VERTICAL:
        return (0, half)
    raise ValueError('invalid direction')
