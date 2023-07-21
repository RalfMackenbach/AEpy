import matplotlib.pyplot as plt
import numpy as np
import matplotlib        as      mpl

from matplotlib import rc

# mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size': 13})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)
# Define parametetric equations for rotated ellipse
def ellipse(t, Rx, Ry, phi):
    x = Rx * np.cos(t) * np.cos(phi) - Ry * np.sin(t) * np.sin(phi)
    y = Rx * np.cos(t) * np.sin(phi) + Ry * np.sin(t) * np.cos(phi)
    return x, y

# make three plots for three different ellipses
c=1
fig, ax = plt.subplots(1, 3, figsize=(c*5/6*6, c*5/6*2.5),sharex=True,sharey=True,tight_layout=True)

# plot at 45 degrees
phi = np.pi / 4

slantedness=np.sqrt(1/2)

# plot the first ellipse
Rx0 = slantedness
print(Rx0**2)
Ry0 = np.sqrt(2-Rx0**2)
t = np.linspace(0, 2 * np.pi, 1000)
x, y = ellipse(t, Rx0, Ry0, phi)
ax[0].plot(x, y, color='k')

# plot the second ellipse
Rx1 = np.sqrt(1)
Ry1 = np.sqrt(2-Rx1**2)
t = np.linspace(0, 2 * np.pi, 1000)
x, y = ellipse(t, Rx1, Ry1, phi)
ax[1].plot(x, y, color='k')

# plot the third ellipse
Ry2 = slantedness
Rx2 = np.sqrt(2-Ry2**2)
print(Rx2**2)
t = np.linspace(0, 2 * np.pi, 1000)
x, y = ellipse(t, Rx2, Ry2, phi)
ax[2].plot(x, y, color='k')

# make plot
# domain is shared, and runs from -1 to 1 in both x and y
for i in range(3):
    ax[i].set_xlim(-1.1, 1.1)
    ax[i].set_ylim(-1.1, 1.1)
    ax[i].set_xticks([-1, 0, 1])
    ax[i].set_yticks([-1, 0, 1])
    ax[i].set_aspect('equal')
    ax[i].set_xlabel(r'$x$')
    ax[i].grid()
    if i == 0:
        ax[i].set_ylabel(r'$y$')
    if i == 0:
        ax[i].set_title(r'$R_x^2 = 1/2$')
    elif i == 1:
        ax[i].set_title(r'$R_x^2 = 1$')
    elif i == 2:
        ax[i].set_title(r'$R_x^2 = 3/2$')

# save and show
plt.savefig('ellipses.png', dpi=1000)

plt.show()