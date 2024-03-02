from utils import *

class ZeroConvergencePlotter:
    """
    Takes a function that for given alpha creates a matrix describing a linear dynamic system and creates a plot for it.
    To create a plot call plotalpha with a line argument for each number in range(0, nlines).
    """

    def __init__(self, f, nlines):
        self.fig = plt.figure()
        self.gs = self.fig.add_gridspec(nlines, 4, right=10, top=10,hspace=1)
        w = 2
        self.Y, self.X = np.mgrid[-w:w:100j, -w:w:100j]
        self.f = f

    def plotalpha(self,alpha, line, type, stable):
        ax1 = self.fig.add_subplot(self.gs[line, 1])
        plot_arbitrary_phase_portrait_in_axis(lambda x: self.f(x, alpha), self.X, self.Y, ax1, 1)
        ax1.set_title(f"alpha = {alpha}")
        sp = np.asarray([0.01, 0])
        solution = scipy.integrate.solve_ivp(lambda x, y: self.f(y, alpha), (0, 100), sp, dense_output=True)
        ax1.plot(solution.y[0], solution.y[1], color="red", label=str(sp))
        sp = np.asarray([2, 0])
        solution = scipy.integrate.solve_ivp(lambda x, y: self.f(y, alpha), (0, 100), sp, dense_output=True)
        dsto0 = np.sqrt(np.sum(np.square(solution.y.astype('float64')), 0))
        print(f"{alpha} {sp} trajectory goes to zero with 1e-6 precision:", np.all(dsto0[1:] - dsto0[:-1] < 1e-6))
        ax1.plot(solution.y[0], solution.y[1], color="red", label=str(sp))
        ax = self.fig.add_subplot(self.gs[line, 3])
        ax.axis('off')
        ax.text(0.5, 0.5, "stable" if stable else "unstable")
        ax3 = self.fig.add_subplot(self.gs[line, 2])
        ax3.axis('off')
        ax3.text(0.5, 0.5, type)
