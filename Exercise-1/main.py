import argparse
from gui import MainGUI

if __name__ == '__main__':
    # Start simulation with specified scenario.json from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="scenario_task1.json",
                        help="Specify scenario file to run")
    parser.add_argument("--algorithm", type=str, default="F",
                        help="Specify pathfinding algorithm. No obstacle avoidance for standard algorithm."
                        "D - Dijkstra Algorithm"
                        "F - Fast Marching Method (default)"
                        "S - Standard Algorithm")
    parser.add_argument("--ignorePedestrians", action='store_true', default=False,
                        help="Toggle pedestrian avoidance."
                             "True - don't avoid pedestrians"
                             "False - avoid pedestrians (default)")
    parser.add_argument("--multiplePedestriansInOneCell", action='store_true', default=False,
                        help="Toggle soft pedestrian avoidance if pedestrian avoidance is turned off doesnt do anything."
                             "True - don't avoid pedestrians"
                             "False - avoid pedestrians (default)")
    args = parser.parse_args()

    # Starts simulation gui
    gui = MainGUI()
    gui.start_gui(args.scenario, args.algorithm, args.ignorePedestrians, args.multiplePedestriansInOneCell)
