import json
import random
from visualization import make_2d_hist
import matplotlib.pyplot as plt
import numpy as np

def make_config_rimea_4(scenario_name,density):
    """
    Make RiMEA 14 scenario config file at scenarios/scenario_name.
    @param scenario_name: The name of the scenario.
    @param density: The pedestrian density to fill the scenario with.
    """
    config1 = {}
    overall_dim = 100
    config1["cell_size"] = [overall_dim, overall_dim]
    config1["iterations"] = 120
    config1["targets"] = [(overall_dim-1, x) for x in range(0, 5)]
    config1["obstacles"] = list((x,5) for x in range(0,overall_dim))
    peds = []
    for x in range(0,overall_dim):
        for y in range(0,5):
            for i in range(int(np.round(random.random()*density*2))):
                peds.append({"position": (x, y), "speed": 1})
    config1["pedestrians"] = peds
    open(scenario_name, "w").write(json.dumps(config1, indent=2))

def make_configs_rimea_7(scenario_name):
    """
    Make RiMEA 7 scenario config file at scenarios/scenario_name.
    Distributing the pedestrian speeds according to the population distribution figure in RiMEA.
    @param scenario_name: The name of the scenario.
    """
    config1 = {}
    overall_dim = 80
    config1["cell_size"] = [overall_dim, overall_dim]
    config1["iterations"] = 400
    config1["targets"] = [(overall_dim-1,x) for x in range(0, overall_dim, 2)]
    config1["obstacles"] = []

    #the function simulating the curve in
    def f(x):
        if x < 20:
            return 0.6+x/20
        if x >= 20:
            return 1.6-(x-20)/60

    peds = [{"position": (0, x), "speed": np.clip(np.random.normal(loc=f(x), scale=0.2*f(x))/2,0.1,1)}
            for x in list(x for x in range(0, overall_dim, 2))]
    config1["pedestrians"] = peds
    def make_hist(vs):
        x_vs = list(x["position"][1] for x in vs)
        y_vs = list(2*x["speed"] for x in vs)
        make_2d_hist(x_vs,y_vs,"age","desired speed")
        plt.clf()

    make_hist(peds)
    open(scenario_name, "w").write(json.dumps(config1, indent=2))