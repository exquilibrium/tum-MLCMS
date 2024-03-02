import math
import numpy


def square(x):
    return x * x

def get_distane_max_metric(x, y):
    return max(abs(x[0]-y[0]), abs(x[1]-y[1]))

def get_distance_euc(x, y):
    """
    :param x: The first 2-tuple.
    :param y: The second 2-tuple.
    :return: The euclidian distance between the two points represented by the 2-tuples.
    """
    return math.sqrt(square(x[0] - y[0]) + square(x[1] - y[1]))


def get_distance_euc_context(x, y, dists):
    """
    A wrapper around get_distance_euc to allow for usage with _dijkstra_inner.
    :param x: The first 2-tuple.
    :param y: The second 2-tuple.
    :param dists: Dictionary containing distances from start for 2-tuples. Only those we are certain of (ie. have been closed)
    :return: The euclidian distance between the two points represented by the 2-tuples.
    """
    return dists[x] + math.sqrt(square(x[0] - y[0]) + square(x[1] - y[1]))


def get_distance_fmm(x, y, dists):
    """
    A function computing the Fast Marching Method node distance update.
    :param x: The first 2-tuple.
    :param y: The second 2-tuple.
    :param dists: Dictionary containing distances from start for 2-tuples. Only those we are certain of (ie. have been closed)
    :return: The euclidian distance between the two points represented by the 2-tuples.
    """
    a, b, c = 0, 0, -1
    a2, b2, c2 = a, b, c

    def differentiate_on_axis(axis_pos, a, b, c, a2, b2, c2):
        vs = list((dists[axis_pos(j)], j) for j in [-1, 1] if (axis_pos(j)) in dists)
        if len(vs) > 0:
            value1, j = max(vs, key=lambda x: x[0])
            a += 1
            b -= 2 * value1
            c += value1 * value1
            if not axis_pos(j * 2) in dists:
                a2 += 1
                b2 -= 2 * value1
                c2 += value1 * value1
            else:
                value2 = dists[axis_pos(j * 2)]
                value21 = 9 / 4
                value22 = (1 / 3) * (4 * value1 - value2)
                a2 += value21
                b2 -= 2 * value21 * value22
                c2 += value1 * value22 * value22
        return a, b, c, a2, b2, c2

    a, b, c, a2, b2, c2 = differentiate_on_axis(lambda j: (y[0] + j, y[1]), a, b, c, a2, b2, c2)
    a, b, c, a2, b2, c2 = differentiate_on_axis(lambda j: (y[0], y[1] + j), a, b, c, a2, b2, c2)
    roots = list(x for x in numpy.roots([a, b, c]) if not numpy.iscomplex(x))
    if a == 0:
        raise Exception("Node must be adjacent to a closed one.")
    if not any(roots):
        max(list(x for x in numpy.roots([a2, b2, c2]) if not numpy.iscomplex(x)))
    r = max(roots)
    return r
