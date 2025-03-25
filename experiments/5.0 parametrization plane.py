import numpy as np
import matplotlib.pyplot as plt
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

# Function definitions
def Z_height_function(x, y):
    return np.sin(x + y)

def compute_gradient(x, y):
    dz_dx = 0.1 * y
    dz_dy = 0.1 * x
    return dz_dx, dz_dy

# Generate grid and compute height values
x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
x, y = np.meshgrid(x, y)
z_height = Z_height_function(x, y)

# Create figure using the default rcParams figure size
fig, ax = plt.subplots()

# Plot contour
contour = ax.contourf(x, y, z_height, levels=25, cmap="magma", rasterized=True)

# Add a colorbar with a thinner width and ticks at -1, 0, 1
cbar = plt.colorbar(contour, ax=ax, fraction=0.02, pad=0.04)
cbar.set_ticks([-1, 0, 1])
cbar.outline.set_visible(True)  # Remove colorbar border

# Remove black borders around the plot
for spine in ax.spines.values():
    spine.set_visible(True)

ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_box_aspect(1)

# Save
plt.savefig("5.0 parametrization plane.pdf", bbox_inches="tight")
plt.show()
