from utils import *
import random

class OneDPlotter:
    """
    For given starting points visualize different one dimensional parametrized dynamical systems.
    Only works for trajectories that do not oscillate.
    The different arrows describing trajectories are placed atop each other in a graph.
    """
    @staticmethod
    def random_color_generator():
        r = random.random()
        g = random.random()
        b = random.random()
        return (r, g, b)

    def __init__(self, sps = None):
        if sps is None:
            self.sps = [0.1,-0.1,2,-2]
        else:
            self.sps = sps

    def plotalpha(self, alpha, f, name):
        w = 2
        Y, X = np.mgrid[-w:w:100j, -w:w:100j]
        ax1 = plt.figure().add_subplot(111)
        def plotone(sp, y):
            solution = scipy.integrate.solve_ivp(lambda x, y: f(y, alpha), (0, 1000), sp, dense_output=True)
            ep = np.clip(solution.y[0][-1],-4.99,4.99)
            ax1.arrow(sp[0], y, ep-sp[0], 0, head_width = 1, head_length = 0.1, length_includes_head = True, label=str(sp[0]), color = self.random_color_generator())
        for x in range(len(self.sps)):
            plotone((self.sps[x],),x)
        ax1.get_yaxis().set_visible(False)
        ax1.set_xlim((-5,5))
        ax1.set_title(f"{name}, alpha = {alpha}")
        ax1.legend()