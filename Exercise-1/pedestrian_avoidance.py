import math
import metrics
import typing


def single_coefficient(offset, max_dist, rate_of_decrease):
    dist = metrics.get_distance_euc((0, 0), offset)
    if dist >= max_dist:
        return 0
    return math.exp(rate_of_decrease / ((dist * dist) - (max_dist * max_dist)))

def get_coefficients(max_dist: float, rate_of_decrease: float = 1, muli: float = 50):
    """
    Normalized coefficients from -max_dist_minus1 to max_dist_minus1 in 2D exponentially decreasing with euclidian distance.
    :param max_dist: The distance at which the coefficients drop to zero, regardless of rate_of_decrease.
    :param rate_of_decrease: How much do the cost multiplicative constants decrease with distance. Positive real.
    :return: A numpy array holding the coefficients.
    """

    max_dist_minus1 = int(max_dist) - 1

    grid = list(((x, y), single_coefficient((x, y),max_dist,rate_of_decrease))
                for x in range(-max_dist_minus1, max_dist_minus1 + 1)
                for y in range(-max_dist_minus1, max_dist_minus1 + 1))
    gs = sum(x[1] for x in grid)
    r = dict((x[0], muli*x[1] / gs) for x in grid if x[1]/gs > 0.01)
    return r


def pedestrian_avoidance_cost_with_coefficients(position: typing.Tuple[int, int], current_pos: typing.Tuple[int, int],
                                                pedestrian_counts: typing.Dict[typing.Tuple[int, int], int],
                                                max_pedestrians_in_one_cell: int, max_dist: float,
                                                rate_of_decrease: float = 1):
    """
    :param position: The position to calculate for.
    :param current_pos: The current position of the pedestrian, so we can ignore that one pedestrian on that position.
    :param pedestrian_counts: The number of pedestrians at a given position.
    :param max_pedestrians_in_one_cell: How many pedestrians fit in one cell.
    :param max_dist: The distance at which the coefficients drop to zero, regardless of rate_of_decrease.
    :param rate_of_decrease: How much do the cost multiplicative constants decrease with distance. Positive real.
    :return: (the cost of moving to the position given the positions of the other pedestrians,
     the matrix of cost multiplicative constants)
    """
    coefficients = get_coefficients(max_dist, rate_of_decrease)
    return pedestrian_avoidance_cost(position, current_pos, pedestrian_counts, max_pedestrians_in_one_cell,
                                     lambda x : coefficients[x])


def pedestrian_avoidance_cost(position: typing.Tuple[int, int], current_pos: typing.Tuple[int, int],
                              pedestrian_counts: typing.Dict[typing.Tuple[int, int], int],
                              max_pedestrians_in_one_cell: int,
                              coefficients: typing.Callable[[typing.Tuple[int, int]], float]):
    """

    :param position: The position to calculate for.
    :param current_pos: The current position of the pedestrian, so we can ignore that one pedestrian on that position.
    :param pedestrian_counts: The number of pedestrians at a given position.
    :param max_pedestrians_in_one_cell: How many pedestrians fit in one cell.
    :param coefficients: Coefficients multiplying the pedestrian counts at given position.
    :return: (the cost of moving to the position given the positions of the other pedestrians,
     the matrix of cost multiplicative constants)
    """
    if position in pedestrian_counts:
        if pedestrian_counts[position] > max_pedestrians_in_one_cell:
            return math.inf

    def cost(offset, c):
        offset_pos = (position[0] + offset[0], position[1] + offset[1])
        count = 0
        if offset_pos in pedestrian_counts:
            count = (pedestrian_counts[offset_pos] -
                     (1 if offset_pos[0] == current_pos[0] and offset_pos[1] == current_pos[1] else 0))
        return count * c

    vs = (cost((v[0], v[1]), coefficients[v]) for v in coefficients)

    return sum(vs)