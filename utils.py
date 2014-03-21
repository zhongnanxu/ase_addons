from ase.utils.eos import EquationOfState
from ase.utils.sjeos import EquationOfStateSJEOS
import numpy as np

def plot_sjeos(self, filename=None, show=None, title='sjeos'):
    """Plot fitted energy curve.

        Uses Matplotlib to plot the energy curve.  Use *show=True* to
        show the figure and *filename='abc.png'* or
        *filename='abc.eps'* to save the figure to a file.

        Note this is a monkey patch of the original because the original
        outputted bad looking graphs."""

    import matplotlib.pyplot as plt
    
    if self.v0 is None:
        self.fit()
        
    if filename is None and show is None:
        show = True

    x = 3.95
    f = plt.figure(figsize=(x * 2.5**0.5, x))
    ax = f.add_subplot(111)
        
    ax.plot(self.v, self.e, marker='o', ls='None')
    x = np.linspace(min(self.v), max(self.v), 100)
    y = self.fit0(x**-(1.0 / 3))
    ax.plot(x, y, color='r')
    try:
        from ase.units import kJ
        ax.set_xlabel(r'Volume [$\AA^3$]')
        ax.set_ylabel(r'Energy [eV]')
        ax.set_title(r'%s: E: %.3f eV, V: %.3f $\AA^3$, B: %.3f GPa' %
                     (title, self.e0, self.v0, self.B / kJ * 1.e24))
    except ImportError:
        ax.set_xlabel(u'volume [L(length)^3]')
        ax.set_ylabel(u'energy [E(energy)]')
        ax.set_title(u'%s: E: %.3f E, V: %.3f L^3, B: %.3e E/L^3' %
                     (title, self.e0, self.v0, self.B))
    f.tight_layout()
    plt.ticklabel_format(useOffset=False)
    if show:
        plt.show()
    if filename is not None:
        f.savefig(filename)

    return f

EquationOfStateSJEOS.plot = plot_sjeos

def plot_eos(self, filename=None, show=None, title='sjeos'):
    if self.eos_string == 'sjeos':
        return EquationOfStateSJEOS.plot(self, filename, show, title)
    else:
        return EquationOfStateASE2.plot(self, filename, show, title)

EquationOfState.plot = plot_eos