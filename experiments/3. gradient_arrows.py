import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles, figsizes, fonts


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



# Function definition
def Z_height_function(x, y):
    return np.sin(x * y)

# Compute gradient
def compute_gradient(x, y):
    dz_dx = 0.1 * y
    dz_dy = 0.1 * x
    return dz_dx, dz_dy

# Generate grid
x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
x, y = np.meshgrid(x, y)
z_height = Z_height_function(x, y)

# Define a grid for arrows at every 0.5 interval
arrow_spacing = 0.5
x_arrows = np.arange(0, 10 + arrow_spacing, arrow_spacing)
y_arrows = np.arange(0, 10 + arrow_spacing, arrow_spacing)
X_arrows, Y_arrows = np.meshgrid(x_arrows, y_arrows)

# Compute gradients for arrow positions
U = np.zeros_like(X_arrows)
V = np.zeros_like(Y_arrows)

for i in range(X_arrows.shape[0]):
    for j in range(X_arrows.shape[1]):
        grad_x, grad_y = compute_gradient(X_arrows[i, j], Y_arrows[i, j])
        U[i, j] = -grad_x  # Negate descent direction
        V[i, j] = -grad_y

# Plot the original function
fig, ax = plt.subplots(figsize=(10, 8))

# Plot contour
contour = ax.contourf(x, y, z_height, levels=50, cmap="plasma", edgecolors="face", rasterized=True)

# Add colorbar
colorbar = plt.colorbar(contour, ax=ax, label='sin(x*y)')

# Plot gradient arrows in white
ax.quiver(X_arrows, Y_arrows, U, V, color='white', angles='xy', scale_units='xy', scale=3)

ax.set_title("Gradient descent with arrows on a grid")
ax.set_xlabel('x')
ax.set_ylabel('y')


plt.savefig(f"3. gradient arrows.pdf", bbox_inches="tight")
plt.show()
