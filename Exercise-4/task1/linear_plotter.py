from utils import *
class LinearPlotter:
    """
    Takes a function that for given alpha creates a matrix describing a linear dynamic system and creates a plot for it.
    To create a plot call plotalpha with a line argument for each number in range(0, nlines).
    """

    def __init__(self, getA, nlines):
        self.fig = plt.figure()
        self.gs = self.fig.add_gridspec(nlines, 4, right=3, top=3,hspace=1)
        w = 3
        self.Y, self.X = np.mgrid[-w:w:100j, -w:w:100j]
        self.getA = getA

    def plotalpha(self,alpha,line,type,stable):
        A = self.getA(alpha)
        ax1 = self.fig.add_subplot(self.gs[line,1])
        plot_phase_portrait_in_axis(A, self.X, self.Y, ax1)
        ax1.set_title(f"alpha = {alpha}")
        ax = self.fig.add_subplot(self.gs[line,3])
        ax.axis('off')
        ax.text(0.5,0.5,"stable" if stable else "unstable")
        ax3 = self.fig.add_subplot(self.gs[line,2])
        ax3.axis('off')
        ax3.text(0.5,0.5,type)
        eigenvals = np.unique(np.linalg.eigvals(A))
        ax2 = self.fig.add_subplot(self.gs[line,0])
        ax2.scatter(np.real(eigenvals),np.imag(eigenvals))
        ax2.set_xlim(-2,2)
        ax2.set_ylim(-1,1)
