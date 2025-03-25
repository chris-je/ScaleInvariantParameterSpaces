import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from tueplots import bundles

# Tueplots to adapt the plot to the paper dimensions/style
plt.rcParams.update(bundles.icml2022(usetex=False, family="Latin Modern Math"))
plt.rcParams.update({
    "figure.figsize": (14 / 2.54, 8 / 2.54),
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5,
    "legend.frameon": True,
})


# select the parametrizations for the fixed weight
constant_w = [1, 2, 6, 10]


for y in constant_w:

    # Create figure and 3D axis using rcParams' figsize (do not override with explicit figsize)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create grid for x and z
    x = np.linspace(0, 10, 200)
    z = np.linspace(0, 10, 200)
    X, Z = np.meshgrid(x, z)

    # Define global limits for function values for consistent coloring
    vmin, vmax = -1, 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('magma')

    # Loop over y-values (every 3 units from 0 to 10)
    # Compute function F = sin(0.3 * Z * X) + sin(0.3 * Z * y) on the xz-plane for constant y
    # F = np.sin(0.3 * (Z * X)) + np.sin(0.3 * Z * y)
    F = np.sin(0.3 * Z * (y + X))

    # Create a constant Y array (plane) with the same shape as X and Z
    Y_plane = np.full_like(X, y)

    # Map the function values to facecolors using the colormap
    facecolors = cmap(norm(F))

    # Plot the flat plane at constant y
    ax.plot_surface(X, Y_plane, Z, facecolors=facecolors, rstride=1, cstride=1, 
                    shade=False, alpha=0.8)

    # Set axis labels and limits
    ax.set_xlabel('w2')
    ax.set_ylabel('w1')
    ax.set_zlabel('w3')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    ax.xaxis.labelpad = 1
    ax.yaxis.labelpad = 1
    ax.zaxis.labelpad = 1
    ax.tick_params(pad=1)


    # plt.xticks(np.arange(0, 10, 5))
    # plt.yticks(np.arange(0, 10, 5))

    # Create a ScalarMappable for the colorbar and add it to the plot
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    # Increase aspect value to make the colorbar thinner, set ticks, and remove the border
    cbar = fig.colorbar(mappable, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_ticks([-1, 0, 1])
    # cbar.outline.set_visible(False)

    plt.savefig(f"5.1 3D contour plane for w={y}.pdf", bbox_inches="tight")
    # plt.show()
