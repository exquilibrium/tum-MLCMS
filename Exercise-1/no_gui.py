from scenario_elements import Scenario, Pedestrian
import matplotlib.pyplot as plt
import os
import json
import math

class Simulation:
    def run(self, config_file, f = None, show_first_f = None, ped_avoidance_coefficients = None, max_peds_in_cell = 1):
        """
        Automatically run the simulation give the amount of iterations
        :param config_file:
        :param f:
        :param show_first_f: Is called on the first frame image.
        :param ped_avoidance_coefficients: Pedestrian avoidance coefficients.
        :param max_peds_in_cell: Max number of pedestrians in cell.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(current_dir, config_file)
        with open(config_file_path, 'r') as config_file:
            self.config = json.load(config_file)
        self.iterations = self.config.get("iterations", 10)
        if self.iterations < 0:
            self.iterations = 0
        cell_size = self.config.get("cell_size", [100, 100])
        target_positions = self.config.get("targets", [])
        obstacle_positions = self.config.get("obstacles", [])
        pedestrian_data = self.config.get("pedestrians", [])
        # ---!!!--- Fix to show bottom 10% of the canvas ---!!!--- #
        cell_size_tmp = math.ceil((10 * cell_size[1]) / 9)
        cell_size_tmp = cell_size_tmp if math.ceil((10 * cell_size[1]) % 9) == 0 else cell_size_tmp - 1
        # ---!!!---
        # Load values into scenario object
        self.scenario = Scenario(cell_size[0], cell_size_tmp, f, ped_avoidance_coefficients, max_peds_in_cell)
        # Add TARGETS based on the configuration
        for target_pos in target_positions:
            self.scenario.grid[target_pos[0], target_pos[1]] = Scenario.NAME2ID['TARGET']
        # Add OBSTACLES based on the configuration
        for obstacle_pos in obstacle_positions:
            self.scenario.grid[obstacle_pos[0], obstacle_pos[1]] = Scenario.NAME2ID['OBSTACLE']
        # Add PEDESTRIANS based on the configuration
        for pedestrian_info in pedestrian_data:
            position, speed = pedestrian_info["position"], pedestrian_info["speed"]
            self.scenario.pedestrians.append(Pedestrian(tuple(position), speed))
        self.scenario.recompute_target_distances("S")
        iterations = self.iterations
        if show_first_f is not None:
            show_first_f(self.scenario.make_image())
        for i in range(self.iterations):
            self.scenario.to_image_save_only()
            self.scenario.update_step(False,i)
            if not self.scenario.pedestrians:
                iterations = i
        return self.scenario, iterations