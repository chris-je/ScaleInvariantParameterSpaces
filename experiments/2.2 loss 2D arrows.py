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
    return np.sin(x * y)

def compute_gradient(x, y):
    dz_dx = 0.1 * y
    dz_dy = 0.1 * x
    return dz_dx, dz_dy

# Generate grid and compute height values
x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
x, y = np.meshgrid(x, y)

# Plot light gray contour lines for x*y = 1, 3, 10
prod = x * y
fig, ax = plt.subplots()
ax.contour(x, y, np.sin(0.5 * x * y), levels=[0], colors='lightgray', linewidths=1)


# Setup a sparse grid for gradient arrows
arrow_spacing = 2.0
x_arrows = np.arange(0, 10 + arrow_spacing, arrow_spacing)
y_arrows = np.arange(0, 10 + arrow_spacing, arrow_spacing)
X_arrows, Y_arrows = np.meshgrid(x_arrows, y_arrows)

# Compute gradients for arrow positions
U = np.zeros_like(X_arrows)
V = np.zeros_like(Y_arrows)
for i in range(X_arrows.shape[0]):
    for j in range(X_arrows.shape[1]):
        grad_x, grad_y = compute_gradient(X_arrows[i, j], Y_arrows[i, j])
        U[i, j] = -grad_x  # descent direction
        V[i, j] = -grad_y

# Plot gradient arrows
ax.quiver(X_arrows, Y_arrows, U, V, color='black', angles='xy', 
          scale_units='xy', scale=1, width=0.006)

# Axis labels and square aspect ratio for the plot area
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_box_aspect(1)

plt.savefig("2.2 loss 2D arrows.pdf", bbox_inches="tight")
plt.show()
