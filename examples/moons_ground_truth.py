# %%

import matplotlib.pyplot as plt
import numpy as np

from mislabeled.datasets.moons import moons_ground_truth_px, moons_ground_truth_pyx

# %%
spread = 0.3

x_edges = np.linspace(-1.75, 2.75, 121)
y_edges = np.linspace(-1.25, 1.75, 81)

x_coords = (x_edges[1:] + x_edges[:-1]) / 2
y_coords = (y_edges[1:] + y_edges[:-1]) / 2

xy = np.stack(np.meshgrid(x_coords, y_coords, indexing="ij"), axis=-1)

p_y_x = moons_ground_truth_pyx(xy.reshape(-1, 2), spread=spread)

plt.figure()
plt.title('ground truth $p(y=1|x)$')
plt.imshow(p_y_x.reshape(120, 80).T, origin="lower", cmap="PiYG")
plt.colorbar()
plt.show()

p_x = moons_ground_truth_px(xy.reshape(-1, 2), spread=spread)

plt.figure()
plt.title('ground truth $p(x)$')
plt.imshow(p_x.reshape(120, 80).T, origin="lower")
plt.colorbar()
plt.show()

# %%
