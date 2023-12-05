__all__ = [
    "plot_grid",
]

import matplotlib.pyplot as plt
import numpy as np


def plot_grid(grid: np.ndarray) -> plt.Axes:
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(grid, cmap="binary")
    plt.grid(True, which="both", color="white", linewidth=1, linestyle="--")
    return ax
