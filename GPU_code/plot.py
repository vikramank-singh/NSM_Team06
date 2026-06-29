#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")      # HPC/headless systems

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

# ------------------------------------------------------------------
# Grid
# ------------------------------------------------------------------

nx = 301
ny = 301
nz = 301

shape = (nz, ny, nx)

# ------------------------------------------------------------------
# Fast I/O
# ------------------------------------------------------------------

t0 = perf_counter()

T = np.loadtxt("T.txt", dtype=np.float64).reshape(shape)
v = np.loadtxt("p.txt", dtype=np.float64).reshape(shape)
I = np.loadtxt("I.txt", dtype=np.float64).reshape(shape)

print(f"Data load time : {perf_counter()-t0:.3f} s")

# ------------------------------------------------------------------
# Derived slices
# ------------------------------------------------------------------

midz = nz // 2
midx = nx // 2

T_mid = T[midz]
v_mid = v[midz]
I_exit = I[-1]

I_xz = np.rot90(I[:, :, midx])
I_yz = np.rot90(I[:, midx, :])

# ------------------------------------------------------------------
# Figure 1
# ------------------------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5),
    constrained_layout=True
)

plots = [
    (T_mid, "Temperature (K) at z=Lz/2"),
    (v_mid, "Pressure at z=Lz/2"),
    (I_exit, "Intensity at z=Lz")
]

for ax, (field, title) in zip(axes, plots):

    im = ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        interpolation="nearest"
    )

    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax)

fig.savefig(
    "midplane_fields.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------------
# Figure 2
# ------------------------------------------------------------------

fig, axes = plt.subplots(
    2, 1,
    figsize=(8, 10),
    constrained_layout=True
)

im = axes[0].imshow(
    I_xz,
    origin="lower",
    aspect="auto",
    cmap="coolwarm",
    interpolation="nearest"
)
axes[0].set_title("XZ Plane")
fig.colorbar(im, ax=axes[0])

im = axes[1].imshow(
    I_yz,
    origin="lower",
    aspect="auto",
    cmap="coolwarm",
    interpolation="nearest"
)
axes[1].set_title("YZ Plane")
fig.colorbar(im, ax=axes[1])

fig.savefig(
    "cross_sections.png",
    dpi=300,
    bbox_inches="tight"
)

print("Plots saved:")
print("  midplane_fields.png")
print("  cross_sections.png")
