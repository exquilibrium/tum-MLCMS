import typing
import random
import scipy.spatial.distance
from PIL import Image, ImageTk
import numpy as np
from shortest_path import get_shortest_path_length, ShortestPathAlg
from metrics import get_distance_euc
from pedestrian_avoidance import get_coefficients, pedestrian_avoidance_cost
import math


class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, desired_speed, avoidance=1):
        self._position = position
        self.start_pos = position
        self._desired_speed = desired_speed
        self.avoidance = avoidance

    @property
    def position(self) -> typing.Tuple[int, int]:
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return scenario.get_neighbors_inner(self.position)

    def update_step(self, scenario, pedestrian_counts, ignore_pedestrians):
        """
        Moves to the cell with the lowest distance to the target with the probability equal to distance divided by speed.
        This takes obstacles or other pedestrians into account.
        Pedestrians cannot occupy the same cell.

        :param scenario: The current scenario instance.
        :param avoidance: How hard to avoid other pedestrians.
        """
        neighbors = self.get_neighbors(scenario)

        def pedestrian_avoidance_cost_for_pos(x):
            return 0 if ignore_pedestrians else pedestrian_avoidance_cost(
                x, self._position, pedestrian_counts, scenario.max_peds_in_cell, scenario.ped_avoidance_coefficients)

        min_cost = (scenario.target_distance_grids[
                        self._position[0], self._position[1]] + self.avoidance * pedestrian_avoidance_cost_for_pos(
            self._position))
        min_cost_pos = self._position

        for pos in neighbors:
            cost = scenario.target_distance_grids[pos] + self.avoidance * 50 * pedestrian_avoidance_cost_for_pos(pos)
            if min_cost > cost:
                min_cost_pos = pos
                min_cost = cost

        if self._position == min_cost_pos:
            return

        # Stochastic movement
        if self.desired_speed / get_distance_euc(self._position, min_cost_pos) < random.random():
            return

        # Paint the previous cell
        scenario.grid[self._position[0], self._position[1]] = Scenario.NAME2ID['VISITED']
        scenario.grid_visited_counts[self._position[0], self._position[1]] += 1
        # Update position
        self._position = min_cost_pos
        if (scenario.grid[self._position] != Scenario.NAME2ID['TARGET']):
            scenario.grid[self._position] = Scenario.NAME2ID['PEDESTRIAN']


class Scenario:
    """
    Scenario for a cellular automaton.
    """
    GRID_SIZE = (500, 500)
    ID2NAME = {
        0: 'EMPTY',
        1: 'TARGET',
        2: 'OBSTACLE',
        3: 'PEDESTRIAN',
        4: 'VISITED'
    }
    NAME2COLOR = {
        'EMPTY': (255, 255, 255),  # White
        'PEDESTRIAN': (255, 0, 0),  # Red
        'TARGET': (0, 0, 255),  # Blue
        'OBSTACLE': (255, 0, 255),  # Magenta
        'VISITED': (208, 208, 208)  # Light Grey
    }
    NAME2ID = {
        ID2NAME[0]: 0,
        ID2NAME[1]: 1,
        ID2NAME[2]: 2,
        ID2NAME[3]: 3,
        ID2NAME[4]: 4
    }

    def __init__(self, width, height, f=None, ped_avoidance_coefficients=None, max_peds_in_cell=1):
        """
        Constructor

        :param width: Width of the cellular automaton
        :param height: Height of the cellular automaton
        :param f: A function taking a scenario that is run every iteration.
        :param ped_avoidance_coefficients: Coefficients for positions for soft pedestrian avoidance None means empty dictionary.
        :param max_peds_in_cell: Hard limit on number of pedestrians in one cell.
        """
        if width < 1 or width > 1024:
            raise ValueError(f"Width {width} must be in [1, 1024].")
        if height < 1 or height > 1024:
            raise ValueError(f"Height {height} must be in [1, 1024].")

        self.width = width
        self.height = height
        self.grid_image = None
        self.grid = np.zeros((width, height))
        self.grid_visited_counts = np.zeros((width, height), dtype=int)
        self.pedestrians = []
        self.target_distance_grids = None
        self.images = []
        self.pedestrians_in_goal = []
        self.f = f
        if ped_avoidance_coefficients == None:
            self.ped_avoidance_coefficients = {}
        else:
            self.ped_avoidance_coefficients = ped_avoidance_coefficients
        self.max_peds_in_cell = max_peds_in_cell

    def recompute_target_distances(self, method):
        """
        Choose algorithm for computing distances.

        :param method:
        :return:
        """
        self.target_distance_grids_fmm = self.update_target_grid_fmm()
        if method == "D":
            self.target_distance_grids = self.update_target_grid_dijkstra()
        elif method == "F":
            self.target_distance_grids = self.update_target_grid_fmm()
        elif method == "S":
            self.target_distance_grids = self.update_target_grid()

        return self.target_distance_grids

    def get_targets(self):
        """
        Get target positions

        :return:
        """
        targets = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append((x, y))
        return targets

    def get_positions(self):
        """
        Get positions

        :return:
        """
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def update_target_grid(self):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        This does not take obstacles into account.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = self.get_targets()
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        targets = np.row_stack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)

        return np.transpose(distances.reshape((self.height, self.width)))

    def is_obstacle(self, x):
        v = self.grid[x]
        return int(v) == int(Scenario.NAME2ID['OBSTACLE'])

    def get_neighbors_inner(self, pos, dist=1):
        """
        Compute all neighbors in a dist cell diagonal neighborhood given position.
        :param pos: Position to get neighbors of.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """

        return [
            (int(x + pos[0]), int(y + pos[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + pos[0] < self.width and 0 <= y + pos[1] < self.height and np.abs(x) + np.abs(y) > 0 and
               not self.is_obstacle((x + pos[0], y + pos[1]))
        ]

    def get_neighbors_inner_no_diag(self, pos):
        """
        Compute all neighbors in a 9 cell neighborhood given position.
        :param pos: Position to get neighbors of.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (int(x + pos[0]), int(y + pos[1]))
            for x in [-1, 0, 1]
            for y in [0]
            if 0 <= x + pos[0] < self.width and 0 <= y + pos[1] < self.height and np.abs(x) + np.abs(y) > 0 and
               not self.is_obstacle((x + pos[0], y + pos[1]))
        ] + [
            (int(x + pos[0]), int(y + pos[1]))
            for x in [0]
            for y in [-1, 0, 1]
            if 0 <= x + pos[0] < self.width and 0 <= y + pos[1] < self.height and np.abs(x) + np.abs(y) > 0 and
               not self.is_obstacle((x + pos[0], y + pos[1]))
        ]

    def update_target_grid_fmm(self):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        This does not take obstacles into account.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = self.get_targets()
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        def makeDistances():
            r = np.full((self.width, self.height), fill_value=np.infty)
            for t in targets:
                dists = get_shortest_path_length(t, lambda x: not self.is_obstacle(x),
                                                 lambda x: self.get_neighbors_inner_no_diag(x),
                                                 ShortestPathAlg.Fast_Marching_Method)
                for pos in dists:
                    r[pos] = min(r[pos], dists[pos])
            return r

        return makeDistances()

    def update_target_grid_dijkstra(self):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        This does not take obstacles into account.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = self.get_targets()
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        def makeDistances():
            r = np.full((self.width, self.height), fill_value=np.infty)
            for t in targets:
                dists = get_shortest_path_length(t, lambda x: not self.is_obstacle(x),
                                                 lambda x: self.get_neighbors_inner(x),
                                                 ShortestPathAlg.Dijkstra_Algorithm)
                for pos in dists:
                    r[pos] = min(r[pos], dists[pos])
            return r

        return makeDistances()

    def get_pedestrian_counts(self):
        pedestrian_counts = {}
        for pedestrian in self.pedestrians:
            if pedestrian.position in pedestrian_counts:
                pedestrian_counts[pedestrian.position] += 1
            else:
                pedestrian_counts[pedestrian.position] = 1
        return pedestrian_counts

    def remove_pedestrians_near_positions(self, positions, dist_to_remove, iteration):
        """
        Removes from self.pedestrians all pedestrians that are less than dist_to_remove diagonal metric from target.
        """
        to_remove = {}
        for i in range(len(self.pedestrians)):
            pedestrian = self.pedestrians[i]
            if any(x for x in positions if
                   max(abs(pedestrian.position[0] - x[0]), abs(pedestrian.position[1] - x[1])) < dist_to_remove):
                to_remove[i] = pedestrian
                self.grid[pedestrian.position] = int(Scenario.NAME2ID['VISITED'])
        self.pedestrians = list(x[1] for x in enumerate(self.pedestrians) if x[0] not in to_remove)
        return list((to_remove[x], iteration) for x in to_remove)

    def update_step(self, ignore_pedestrians, iteration):
        """
        Updates the position of all pedestrians.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        pedestrians_disappear_when_at_target = True
        pedestrian_counts = self.get_pedestrian_counts()
        targets = self.get_targets()
        for pedestrian in self.pedestrians:
            old_pos = pedestrian.position
            pedestrian.update_step(self, pedestrian_counts, ignore_pedestrians)
            pedestrian_counts[old_pos] -= 1
            if pedestrian.position in pedestrian_counts:
                pedestrian_counts[pedestrian.position] += 1
            else:
                pedestrian_counts[pedestrian.position] = 1

        if self.f is not None:
            self.f(self)

        if pedestrians_disappear_when_at_target:
            self.pedestrians_in_goal = self.pedestrians_in_goal + self.remove_pedestrians_near_positions(targets, 3,
                                                                                                         iteration)

    def cell_to_color(self, x, y):
        _id = self.grid[x, y]
        if (_id == int(Scenario.NAME2ID['VISITED'])):
            value = 215 - min(85, self.grid_visited_counts[x, y] * 10)
            return (value, value, value)
        else:
            return Scenario.NAME2COLOR[Scenario.ID2NAME[_id]]

    @staticmethod
    def pedestrians_to_color(stack_counts):
        return (255 - min(50, stack_counts * 10), 30 - min(stack_counts * 10, 30), 30 - min(stack_counts * 10, 30))

    def target_grid_to_image(self, canvas, canvas_image):
        """
        Creates a colored image based on the distance to the target stored in
        self.target_distance_gids.
        :param canvas: the canvas that holds the image.
        :param canvas_image: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                target_distance = self.target_distance_grids[x][y]
                if math.isfinite(target_distance):
                    pix[x, y] = (max(0, min(255, int(13 * target_distance) - 2 * 255)),
                                 max(0, min(255, int(13 * target_distance) - 1 * 255)),
                                 max(0, min(255, int(13 * target_distance) - 0 * 255)))
                else:
                    pix[x, y] = (255, 0, 255)
        pedestrian_count = self.get_pedestrian_counts()
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            pix[x, y] = self.pedestrians_to_color(pedestrian_count[pedestrian.position])
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(canvas_image, image=self.grid_image)

    def make_image(self):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterward, separately.
        :param canvas: the canvas that holds the image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                pix[x, y] = self.cell_to_color(x, y)
            pedestrian_count = self.get_pedestrian_counts()
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            pix[x, y] = self.pedestrians_to_color(pedestrian_count[pedestrian.position])
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        return im

    def to_image_save_only(self):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterward, separately.
        :param canvas: the canvas that holds the image.
        """
        self.images.append(self.make_image())

    def to_image(self, canvas, canvas_image):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterward, separately.
        :param canvas: the canvas that holds the image.
        :param canvas_image: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                pix[x, y] = self.cell_to_color(x, y)
            pedestrian_count = self.get_pedestrian_counts()
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            pix[x, y] = self.pedestrians_to_color(pedestrian_count[pedestrian.position])
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(canvas_image, image=self.grid_image)

    def to_image_visualize_speed(self, canvas, canvas_image):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterward, separately.
        :param canvas: the canvas that holds the image.
        :param canvas_image: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        pedestrians_by_pos = {}
        for pedestrian in self.pedestrians:
            if pedestrian.position in pedestrians_by_pos:
                pedestrians_by_pos[pedestrian.position].append(pedestrian)
            else:
                pedestrians_by_pos[pedestrian.position] = [pedestrian]
        for pedestrian_pos in pedestrians_by_pos:
            pix[pedestrian_pos] = tuple(int(x) for x in (np.asarray(self.NAME2COLOR['PEDESTRIAN']) *
                                                         np.average(list(x.desired_speed for x in
                                                                         pedestrians_by_pos[pedestrian_pos]))))
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(canvas_image, image=self.grid_image)
