import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
 

def latexify(ticklabels):
    """Manually set LaTeX format for tick labels."""
    return [r"$" + str(label) + "$" for label in ticklabels]


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 9

notd = 3
ndim = 3

time = []
traj = []
lyap = []
uDeep = []
uNum = []

for ii in range(notd):
    data = np.genfromtxt('dOTD_tst' + str(ii+1) + '.out')
    time.append(data[:,0])
    traj.append(data[:,1:ndim+1])
    lyap.append(data[:,ndim+1:ndim+2])
    uDeep.append(data[:,ndim+2:2*ndim+2])
    uNum.append(data[:,2*ndim+2::])

x = traj[0][:,0]
y = traj[0][:,1]
z = traj[0][:,2]
ell = lyap[0][:].squeeze()

sc = 0.1
colors = ["#1b9e77", "#d95f02", "#7570b3"]


fig = plt.figure(figsize=(1.7,1.6))

ax = fig.add_axes([-0.01, 0.11, 1.0, 0.9], projection='3d')
ax.grid(False)
ax.axis('off')
ax.view_init(elev=15, azim=45)

cbarticks = [-1e-2, 0, 1e-2]
im = ax.scatter(x, y, z, c=ell, s=0.1, \
                vmin=cbarticks[0], vmax=cbarticks[-1], cmap='RdBu')
cax = fig.add_axes([0.1, 0.12, 0.8, 0.03])
cbar = plt.colorbar(im, cax=cax, orientation="horizontal", ticks=cbarticks)
#cbar.set_ticklabels(latexify(cbarticks))
cbar.set_ticklabels(latexify(["-10^{-2}", "0", "10^{-2}"]))
cbar.ax.tick_params(length=2)


inds = [200, 300]
for ind in inds:
    for ii, mode in enumerate(uDeep):
        arrow_prop_dict = dict(mutation_scale=5,  arrowstyle='->', \
                               shrinkA=0, shrinkB=0)
        a = Arrow3D([x[ind], sc*mode[ind,0]+x[ind]], \
                    [y[ind], sc*mode[ind,1]+y[ind]], \
                    [z[ind], sc*mode[ind,2]/2+z[ind]], \
                    **arrow_prop_dict, color=colors[ii], zorder=100)
        ax.add_artist(a)

for aa in [ax.xaxis, ax.yaxis, ax.zaxis]:
    aa.pane.set_edgecolor("black")
    aa.pane.set_alpha(0)
    aa.pane.fill = False

arrow_prop_dict = dict(mutation_scale=10, arrowstyle='->', \
                       shrinkA=0, shrinkB=0, color='k')
arx = Arrow3D([0, 0.6], [0, 0], [0, 0], **arrow_prop_dict)
ary = Arrow3D([0, 0], [0, 0.6], [0, 0], **arrow_prop_dict)
arz = Arrow3D([0, 0], [0, 0], [0, 0.19], **arrow_prop_dict)
for a in [arx, ary, arz]:
    ax.add_artist(a)

a = Arrow3D([0, 0], [0, 0], [0.11, 0.19], **arrow_prop_dict, zorder=100)
ax.add_artist(a)

ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_zlim(0, 0.15)

ax.text(0.0, 0.0, -0.023, r'$0$', ha='center', va='center')
ax.text(0.64, 0, 0, r'$x$', ha='center', va='center')
ax.text(0, 0.64, 0, r'$y$', ha='center', va='center')
ax.text(0, 0, 0.203, r'$z$', ha='center', va='center')


plt.savefig('cart.pdf')
plt.close()



