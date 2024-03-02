import no_gui
from scenario_maker import make_configs_rimea_7, make_config_rimea_4
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import visualization
from scipy.stats import ttest_ind
from no_gui import Simulation
from metrics import get_distance_euc


def show_desired_actual_speed_hexbin(sn):
    s, i = Simulation().run(sn)
    for i in range(0, i, 5):
        IPython.display.display(s.images[i])
    visualization.make_2D_hist_of_speed(s, i)


def Welch_statistical_test_against_RiMEA_pop_speeds(scenario_name):
    """
    Loads scenario from a file. Simulating it using no_gui.Simulation.
    First it shows the simulation. Then it creates a diagram comparing the desired and actual speeds.
    Then it shows the Welch's test statistic where Hypothesis 0 is that the two distributions have the same mean.
    Then it plots the manually parametrized simulated approximation of the RiMEA distribution.
    @param scenario_name: File to load config from.
    @return:
    """
    s, i = Simulation().run(scenario_name)
    for i in range(0, i, 5):
        IPython.display.display(s.images[i])
    _, actual_speeds, ped_starting_pos = visualization.get_pedestrian_speeds(s, i)
    visualization.make_2D_hist_of_speed(s, i)

    def f(x, y):
        return np.random.normal(x, y)

    manual_approximation_x = [5, 10, 20, 30, 40, 50, 60, 80]
    manual_approximation = [0.74, f(1.2, 0.3), f(1.6, 0.3), f(1.52, 0.3), f(1.5, 0.3), f(1.4, 0.3), f(1.3, 0.25), 0.69]
    print(ttest_ind(2 * actual_speeds, manual_approximation, equal_var=False))
    visualization.make_2d_hist(manual_approximation_x, manual_approximation, "age", "speed")


class MeasurementFunctor:
    """
    Used for measuring pedestrian speeds in 2x2 grids at positions given in init.
    """
    def __init__(self, positions = None):
        """
        @param positions: Positions of the left top corner of the 2x2 grid to measure at.
        """
        if positions is None:
            positions = [(44, 2), (49, 2), (49, 4)]
        self.measure_points = len(positions)
        self.measure_point_peds = list([] for pos in positions)
        self.measure_point_peds_old_poss = list([] for pos in positions)
        offsets = list((x, y) for x in range(0, 2) for y in range(0, 2))
        def getPoss(x, y):
            return set((poss[0] + x, poss[1] + y) for poss in offsets)

        self.measure_point_poss = list(getPoss(pos[0], pos[1]) for pos in positions)
        self.measure_point_avg_speeds = list([] for pos in positions)

    def run(self, scenario):
        def get_avg_speed(point_peds, point_peds_old_poss):
            speeds = []
            for i in range(len(point_peds)):
                d = scenario.target_distance_grids_fmm[point_peds_old_poss[i]] - scenario.target_distance_grids_fmm[point_peds[i].position]
                speeds.append(max(0,d))
            return np.average(speeds)

        def get_new_peds(point_poss):
            return list(x for x in scenario.pedestrians if x.position in point_poss)

        for x in range(self.measure_points):
            new_measure_point_peds = get_new_peds(self.measure_point_poss[x])
            if len(self.measure_point_peds[x]) > 0:
                self.measure_point_avg_speeds[x].append(
                    get_avg_speed(self.measure_point_peds[x], self.measure_point_peds_old_poss[x]))
            self.measure_point_peds[x] = new_measure_point_peds
            self.measure_point_peds_old_poss[x] = list(x.position for x in new_measure_point_peds)


def get_first_pedestrian_speed(sn):
    s, i = Simulation().run(sn)
    return s.pedestrians_in_goal[0][1]
