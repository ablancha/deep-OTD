import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 9

ndim = 6
data = np.genfromtxt('dOTD_tst1.out')

xticks = [900, 1100, 1300]
yticks = [[0.7, 0.8, 0.9, 1],
          [-0.2, 0, 0.2, 0.4],
          [-0.5, 0, 0.5],
          [-1, -0.5, 0],
          [-0.5, 0, 0.5],
          [-0.5, 0, 0.5, 1]]

def latexify(ticklabels):
    """Manually set LaTeX format for tick labels."""
    return [r"$" + str(label) + "$" for label in ticklabels]
         
for ii in range(ndim):
    fig = plt.figure(figsize=(2.2,1.3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    ax = plt.axes()
    plt.plot(data[:,0], data[:,ii+1], 'k-', linewidth=0.75)
    plt.xlabel('$t$')
    plt.ylabel('$z_{' + str(ii+1) + '}$')
    plt.xlim(xticks[0], xticks[-1])
    plt.ylim(yticks[ii][0], yticks[ii][-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks[ii])
    ax.set_xticklabels(latexify(xticks))
    ax.set_yticklabels(latexify(yticks[ii]))
    ax.yaxis.set_label_coords(-0.2, 0.5)
    ax.tick_params(direction='in', length=2)
    plt.savefig('traj' + str(ii+1) + '.pdf')


