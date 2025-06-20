import numpy as np
import torch
import matplotlib.pyplot as plt

# Step 1: Define polar grid
grid_size = 49
x = torch.linspace(-1.0, 1.0, grid_size)
X, Y = torch.meshgrid(x, x, indexing='ij')
R = torch.sqrt(X**2 + Y**2)
Phi = torch.atan2(Y, X)

# Step 2: Define circular harmonics
max_j = 3  # radial levels (columns)
max_k = 5  # angular frequencies (rows from -5 to +5)
rings = [0.25 * (j + 1) for j in range(max_j + 1)]
sigma = 0.15

# Step 3: Organize filters in 2D (k, j)
filter_grid = [[None for _ in range(max_j + 1)] for _ in range(2 * max_k + 1)]
labels = [[None for _ in range(max_j + 1)] for _ in range(2 * max_k + 1)]

for j, r0 in enumerate(rings):
    radial = torch.exp(-((R - r0) ** 2) / (2 * sigma ** 2))
    for k in range(max_k + 1):
        angular = torch.exp(1j * k * Phi)
        psi = radial * angular

        # real part -> +k row
        row_pos = max_k + k
        filter_grid[row_pos][j] = psi.real
        labels[row_pos][j] = f"j={j}, k={k}"

        if k > 0:
            # imaginary part -> -k row
            row_neg = max_k - k
            filter_grid[row_neg][j] = psi.imag
            labels[row_neg][j] = f"j={j}, k={-k}"

# Step 4: Visualize grid of filters with padded titles
fig, axs = plt.subplots(2 * max_k + 1, max_j + 1, figsize=(4, 14))  # Was (12, 14) or more


for i in range(2 * max_k + 1):
    for j in range(max_j + 1):
        ax = axs[i, j]
        filt = filter_grid[i][j]
        if filt is not None:
            ax.imshow(filt.numpy(), cmap='gray', origin='lower')
            ax.set_title(labels[i][j], fontsize=9, pad=5)  # <-- Increased pad here
        ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=0.6, wspace=0.02)

plt.show()

