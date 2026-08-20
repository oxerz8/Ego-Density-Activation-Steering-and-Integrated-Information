# %%
# ============================================================
# EGO-VECTOR ROBUSTNESS TEST
# ============================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pyphi

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

ALPHAS = [-10,-6,-2,0,2,6,10,14,18,20]
N_BOOTSTRAPS = 5
TEST_STATE = (1,0,1)

# ------------------------------------------------------------
# Reuse the original ego-pair activations
#
# 'high' and 'low' must already exist from the original
# ego-vector construction.
# ------------------------------------------------------------

print(
    "Available ego examples:",
    len(high)
)

# ------------------------------------------------------------
# Fixed IIT connectivity
# ------------------------------------------------------------

cm = np.ones(
    (N_NODES, N_NODES),
    dtype=int
)

# ------------------------------------------------------------
# Make bootstrap ego vectors
# ------------------------------------------------------------

def make_bootstrap_ego(seed):

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(high),
        size=len(high)
    )

    v = (
        high[indices].mean(0)
        - low[indices].mean(0)
    )

    v = v / v.norm()

    return v


ego_vectors = [
    make_bootstrap_ego(1000 + i)
    for i in range(N_BOOTSTRAPS)
]

# Include original ego vector
ego_vectors.insert(
    0,
    ego_vector
)

print(
    "\nConstructed",
    len(ego_vectors),
    "ego vectors."
)

# ------------------------------------------------------------
# Φ calculation
# ------------------------------------------------------------

def calculate_phi(
    steering_vector,
    alpha
):

    next_activation = propagate(
        steering_vector,
        alpha
    )

    coordinates = (
        next_activation - baseline
    ) @ basis.T

    tpm = torch.sigmoid(
        coordinates
    ).cpu().numpy()

    tpm = np.clip(
        tpm,
        1e-5,
        1 - 1e-5
    )

    substrate = pyphi.Substrate(
        tpm=tpm,
        cm=cm
    )

    result = pyphi.analyze(
        substrate,
        TEST_STATE,
        formalism="IIT_3_0",
        compute="sia"
    )

    return float(result.phi)

# ============================================================
# RUN
# ============================================================

results = []

for vector_id, vector in enumerate(
    ego_vectors
):

    print(
        f"\n=============================="
    )

    print(
        f"EGO VECTOR {vector_id}"
    )

    print(
        f"=============================="
    )

    for alpha in ALPHAS:

        print(
            f"α = {alpha:+}"
        )

        phi = calculate_phi(
            vector,
            alpha
        )

        results.append({
            "ego_vector": vector_id,
            "alpha": alpha,
            "phi": phi
        })

        print(
            "Φ =",
            phi
        )

# ============================================================
# RESULTS
# ============================================================

df_boot = pd.DataFrame(
    results
)

print(
    "\n=============================="
)

print(
    "BOOTSTRAP RESULTS"
)

print(
    "=============================="
)

print(
    df_boot.to_string(
        index=False
    )
)

# ============================================================
# MEAN ± STD ACROSS EGO VECTORS
# ============================================================

summary = (
    df_boot
    .groupby("alpha")["phi"]
    .agg(
        mean="mean",
        std="std"
    )
    .reset_index()
)

print(
    "\n=============================="
)

print(
    "MEAN Φ ACROSS EGO VECTORS"
)

print(
    "=============================="
)

print(
    summary.to_string(
        index=False
    )
)

# ============================================================
# FIND PEAK OF MEAN CURVE
# ============================================================

peak = summary.loc[
    summary["mean"].idxmax()
]

print(
    "\n=============================="
)

print(
    "MEAN CURVE PEAK"
)

print(
    "=============================="
)

print(
    "α:",
    peak["alpha"]
)

print(
    "Mean Φ:",
    peak["mean"]
)

print(
    "SD:",
    peak["std"]
)

# ============================================================
# CHECK FOR INVERTED-U
# ============================================================

peak_idx = summary["mean"].idxmax()

if (
    peak_idx > 0
    and peak_idx < len(summary)-1
):

    left = summary.iloc[
        peak_idx - 1
    ]["mean"]

    peak_value = summary.iloc[
        peak_idx
    ]["mean"]

    right = summary.iloc[
        peak_idx + 1
    ]["mean"]

    if (
        peak_value > left
        and peak_value > right
    ):
        print(
            "\n✓ Mean trajectory has a local inverted-U."
        )
    else:
        print(
            "\n✗ Mean trajectory does not have "
            "a local inverted-U."
        )

# ============================================================
# PLOT INDIVIDUAL VECTORS
# ============================================================

plt.figure(
    figsize=(8,5)
)

for vector_id in df_boot[
    "ego_vector"
].unique():

    subset = df_boot[
        df_boot["ego_vector"] == vector_id
    ]

    label = (
        "original"
        if vector_id == 0
        else f"bootstrap {vector_id}"
    )

    plt.plot(
        subset["alpha"],
        subset["phi"],
        marker="o",
        alpha=0.6,
        label=label
    )

plt.xlabel(
    "Steering strength α"
)

plt.ylabel(
    "IIT Φ"
)

plt.title(
    "Robustness across ego-vector constructions"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "ego_vector_robustness.png",
    dpi=300
)

plt.show()

# ============================================================
# SAVE
# ============================================================

df_boot.to_csv(
    "ego_vector_robustness.csv",
    index=False
)

summary.to_csv(
    "ego_vector_robustness_summary.csv",
    index=False
)

print(
    "\nSaved robustness results."
)
