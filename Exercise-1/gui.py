import math
import sys
import json
import time
import tkinter
from tkinter import Button, Canvas, Label, StringVar, filedialog, ttk
from scenario_elements import Scenario, Pedestrian
import os
import visualization 
from pedestrian_avoidance import get_coefficients
class MainGUI:
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def __init__(self):
        self.win = None  # The Tk application object created by instantiating Tk()
        self.canvas = None  # Create a canvas widget for drawing graphics.
        self.canvas_image = None  # Draw an image. Returns the item id.

        self.config = None  # Simulation parameters of the scenario
        self.algorithm = "F"  # Current algorithm for calculating target distances
        self.ignorePedestrians = False # True to avoid pedestrians
        self.multiplePedestriansInOneCell= False # Different pedestrian avoidance

        self.background_type = 0  # Background type
        self.scenario = None  # Information about scenario (pedestrians, obstacles, targets)
        self.iterations = 0  # Amount of simulations steps to run in total
        self.start_iterations = 0
        self.id_text_finished = None  # Reference to text widget
        self.running = False

    # Functions for buttons
    def load_scenario(self, iteration_label):
        """
        Initiate dialogue window to open scenario file

        :param iteration_label: Label counting remaining iterations.
        :return:
        """
        current_dir = os.path.dirname(os.path.abspath(__file__)) + "/scenarios"
        filepath = filedialog.askopenfilename(
            title='Open a scenario file',
            initialdir=current_dir,
            filetypes=[("JSON", '*.json')],
        )
        if filepath[-5:] == ".json":
            with open(filepath, 'r') as config_file:
                self.config = json.load(config_file)
            self.win.title("Cellular Automaton GUI - " + os.path.split(filepath)[1])
        self.load_scenario_from_config(iteration_label)

    def start_scenario(self, iteration_label):
        """
        Starts the simulation and runs it for the specified iterations

        :param iteration_label: Label counting remaining iterations.
        :return:
        """
        self.running = True
        self.run_simulation(iteration_label)

    def step_scenario_and_pause(self,iteration_label):
        self.running = False
        time.sleep(0.05)
        self.step_scenario(iteration_label)

    def step_scenario(self, iteration_label):
        """
        Moves the simulation forward by one step, and visualizes the result.

        :param iteration_label: Label counting remaining iterations.
        :return:
        """
        just_ended = False
        if self.iterations <= 0 or not self.scenario.pedestrians:
            return

        self.iterations -= 1
        iteration_label.set("Iteration: " + str(self.iterations))

        self.scenario.update_step(self.ignorePedestrians, self.start_iterations - self.iterations)
        self.scenario.to_image_method(self.canvas, self.canvas_image)

        if self.iterations == 0 or not self.scenario.pedestrians:  # check if pedestrian list is empty
            self.canvas.delete(self.id_text_finished)
            self.id_text_finished = self.canvas.create_text(250, 250, fill="black",
                                                            font=("Times New Roman", 35, "bold italic"),
                                                            text="Simulation Finished")
            visualization.make_2D_hist_of_speed(self.scenario, self.start_iterations)

    def reset_scenario(self, iteration_label):
        """
        Resets the currently loaded scenario.

        :param iteration_label: Label counting remaining iterations.
        :return:
        """
        self.load_scenario_from_config(iteration_label)

    # GUI Functions
    def exit_gui(self):
        """
        Close the GUI.
        """
        sys.exit()

    def start_gui(self, config_file, algorithm, ignore, multiplePedestriansInOneCell):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.

        :param config_file: Starting configuration of a scenario. Pedestrians should be a dictionary of pair of integers for position and
        speed, speed being a float in range from 0.1 to 1. Other values should be pairs of integers for position.
        :param algorithm: The algorithm to calculate target distances .
        :param pedestrians: Toggle pedestrian avoidance.
        :return:
        """
        # 1.) Create GUI
        # Create tkinter window
        window_size = [800, 600]
        self.win = tkinter.Tk()
        self.win.geometry(f"{window_size[0]}x{window_size[1]}")
        self.win.title('Cellular Automata GUI')
        # Create canvas
        self.canvas = Canvas(self.win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])
        self.canvas_image = self.canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        self.canvas.pack()

        # 2.) Load simulation parameters
        # CLI arguments
        self.algorithm = algorithm
        self.ignorePedestrians = ignore
        self.multiplePedestriansInOneCell = multiplePedestriansInOneCell
        # Scenario configuration from the JSON file
        current_dir = os.path.dirname(os.path.abspath(__file__)) + "/scenarios"
        config_file_path = os.path.join(current_dir, config_file)
        with open(config_file_path, 'r') as config_file:
            self.config = json.load(config_file)
            self.win.title("Cellular Automaton GUI - " + os.path.split(config_file_path)[1])

        # GUI Label to count remaining iterations
        iteration_label = StringVar()
        lbl = Label(self.win, textvariable=iteration_label, fg='red')
        lbl.place(x=620, y=15)

        # 3.) Load scenario into simulation
        self.load_scenario_from_config(iteration_label)

        # GUI Buttons
        btn = Button(self.win, text='Load simulation', command=lambda: self.load_scenario(iteration_label))
        btn.place(x=20, y=10)
        btn = Button(self.win, text='Start simulation', command=lambda: self.start_scenario(iteration_label))
        btn.place(x=170, y=10)
        btn = Button(self.win, text='Step simulation', command=lambda: self.step_scenario_and_pause(iteration_label) )
        btn.place(x=320, y=10)
        btn = Button(self.win, text='Reset simulation', command=lambda: self.reset_scenario(iteration_label))
        btn.place(x=470, y=10)
        btn = Button(self.win, text='Change background', command=lambda: self.change_background())
        btn.place(x=20, y=550)

        # Put everything on the display and respond to user input until the program terminates.
        self.win.mainloop()

    def change_background(self):
        """
        This function allows you to change the background type of the application.

        :return:
        """
        if self.background_type == 0:
            self.scenario.to_image_method = self.scenario.target_grid_to_image
            self.background_type = 1
            self.scenario.to_image_method(self.canvas, self.canvas_image)
        else:
            self.scenario.to_image_method = self.scenario.to_image
            self.background_type = 0
            self.scenario.to_image_method(self.canvas, self.canvas_image)

    # Important functions
    def load_scenario_from_config(self, iteration_label):
        """
        Loads the scenario parameters from the config file.

        :param iteration_label: Label counting remaining iterations
        :return:

        Notes:
        # Scenario file format (check scenario_task1.json for example):
            iterations: iter: int                       total amount of simulation steps
            cell_size: [height: int, width: int]        simulation area of the scenario
            target_positions: [[x: int, y: int]]        list of target positions
            obstacle_positions: [[x: int, y: int]]      list of obstacle positions
            pedestrian_data: [{                         list of pedestrian data
                "positions": [x: int, y: int],              pedestrian position
                "speed": s: float                           pedestrian speed
                }]
        """
        # 0.) Reset canvas
        self.canvas.delete(self.id_text_finished)

        # 1.) Load in scenario file
        self.start_iterations = self.config.get("iterations", 10)
        if self.start_iterations < 0:
            self.start_iterations = 0
        self.iterations = self.start_iterations
        iteration_label.set("Iteration: " + str(self.iterations))
        cell_size = self.config.get("cell_size", [100, 100])
        target_positions = self.config.get("targets", [])
        obstacle_positions = self.config.get("obstacles", [])
        pedestrian_data = self.config.get("pedestrians", [])
        # ---!!!--- Fix to show bottom 10% of the canvas ---!!!--- #
        cell_size_tmp = math.ceil((10 * cell_size[1]) / 9)
        cell_size_tmp = cell_size_tmp if math.ceil((10 * cell_size[1]) % 9) == 0 else cell_size_tmp - 1
        # ---!!!---
        if self.multiplePedestriansInOneCell:
            self.scenario = Scenario(cell_size[0], cell_size_tmp,
                                     ped_avoidance_coefficients=get_coefficients(5, 1000, 1 / 10), max_peds_in_cell=6)
        else:
            self.scenario = Scenario(cell_size[0], cell_size_tmp)
        self.scenario.to_image_method = self.scenario.to_image if self.background_type == 0 else self.scenario.target_grid_to_image

        # 2.) Load values into scenario object
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
        # Show pedestrians and targets
        self.scenario.target_distance_grids = self.scenario.recompute_target_distances(self.algorithm)
        self.scenario.to_image_method(self.canvas, self.canvas_image)

    def run_simulation(self, iter_var):
        """
        Automatically run the simulation the given amount of iterations.

        :param iter_var: Label counting remaining iterations.
        :return:
        """
        while self.iterations > 0 and self.running:
            if not self.scenario.pedestrians:
                break
            self.step_scenario(iter_var)
            self.win.update()
            time.sleep(0.01)

        if not self.scenario.pedestrians:  # check if pedestrian list is empty
            self.canvas.delete(self.id_text_finished)
            self.id_text_finished = self.canvas.create_text(250, 250, fill="black",
                                                            font=("Times New Roman", 35, "bold italic"),
                                                            text="Simulation Finished")