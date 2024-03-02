import typing
import enum
from efficient_heap import EfficientHeap
import metrics


class GridWithObstacles:
    """
    """    
    def __init__(self, can_move, get_neighbors):
        self.can_move = can_move
        self.get_neighbors = get_neighbors

    def get_adjacent(self, x):
        return list(v for v in self.get_neighbors(x) if self.can_move(v))


class ShortestPathAlg(enum.Enum):
    Dijkstra_Algorithm= 1
    Fast_Marching_Method = 2


def get_shortest_path_length(start: typing.Tuple[int, int],
                             can_move: typing.Callable[[typing.Tuple[int, int]], bool],
                             get_adjacent: typing.Callable[
                                 [typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]]],
                             algorithm: ShortestPathAlg) -> (
        typing.Dict[typing.Tuple[int, int], float]):
    """
    A wrapper around a FMM algorithm and dijkstra's algorithm with euclidian distance.
    :param start: The starting position.
    :param can_move: A function taking an object of the same type as start and determining whether it is accessible.
    :param get_adjacent: A function taking an object of the same type as start and returning objects next to it.
    They must be of the same type.
    :param algorithm: Determine the sort of algorithm to use.
    :return: A dictionary that contains distance from start for each object reachable from start.
    """
    grid = GridWithObstacles(can_move, get_adjacent)
    if algorithm == ShortestPathAlg.Dijkstra_Algorithm:
        r = _dijkstra_inner(start, grid.get_adjacent, metrics.get_distance_euc_context)
    else:
        r = _dijkstra_inner(start, grid.get_adjacent, metrics.get_distance_fmm)
    return r

def _dijkstra_inner(start: typing.Tuple[int, int],
                    get_neighbors: typing.Callable[[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]]],
                    get_distance: typing.Callable[[typing.Tuple[int, int], typing.Tuple[int, int],
                                                   typing.Dict[typing.Tuple[int, int], float]], float]) -> (
        typing.Dict[typing.Tuple[int, int], float]):
    """
    An implementation of the fast marching method.
    :param start: The starting position.
    :param can_move: A function taking a position and determining whether it is accessible.
    :param get_adjacent: A function returning objects next to a position.
    They must be of the same type.
    :return: A dictionary that contains distance from start for each position reachable from start.
    """
    frontier = EfficientHeap()
    seen = {}
    distances_from_start: typing.Dict[typing.Tuple[int, int], float] = dict()
    frontier.add(start, 0)
    seen[start] = 0
    distances_from_start[start] = 0
    closed: typing.Dict[typing.Tuple[int, int], float] = dict()
    while len(frontier) > 0:
        (v, d) = frontier.pop()
        closed[v] = d
        for neigh in get_neighbors(v):
            if neigh in closed:
                continue
            new_d = get_distance(v, neigh, closed)
            if neigh in seen.keys():
                old_d = distances_from_start[neigh]
                if new_d < old_d:
                    frontier.decrease(seen[neigh], new_d)
                    distances_from_start[neigh] = new_d
            else:
                seen[neigh] = frontier.add(neigh, new_d)
                distances_from_start[neigh] = new_d
    return distances_from_start
