import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles

# Tueplots to adapt the plot to the paper dimensions/style
plt.rcParams.update(bundles.icml2022(usetex=False, family="Latin Modern Math"))
plt.rcParams.update({
    "figure.figsize": (7 / 2.54, 4 / 2.54),
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5,
    "legend.frameon": True,
})

# Define x values
x = np.linspace(-2, 2, 400)
# For negative x, set the value to 0
y = np.where(x > 0, x, 0)

fig, ax = plt.subplots()
ax.plot(x, y, color="#231152")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)

plt.savefig("1.1 ReLU.pdf", bbox_inches="tight")
plt.show()
