# %%
# ============================================================
# TRUE SHAM-EGO CONTROL
# ============================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pyphi

ALPHAS = [-10,-6,-2,0,2,6,10,14,18,20]
N_SHAMS = 5
TEST_STATE = (1,0,1)

cm = np.ones(
    (N_NODES, N_NODES),
    dtype=int
)

# ============================================================
# BUILD SHAM VECTORS
#
# Randomly split ALL examples into two groups.
# This destroys the self-vs-objective semantic distinction.
# ============================================================

all_activations = torch.cat(
    [high, low],
    dim=0
)

N_EXAMPLES = all_activations.shape[0]
GROUP_SIZE = N_EXAMPLES // 2


def make_sham_vector(seed):

    rng = np.random.default_rng(seed)

    indices = rng.permutation(
        N_EXAMPLES
    )

    group_a = all_activations[
        indices[:GROUP_SIZE]
    ]

    group_b = all_activations[
        indices[GROUP_SIZE:GROUP_SIZE*2]
    ]

    v = (
        group_a.mean(0)
        - group_b.mean(0)
    )

    v = v / v.norm()

    return v


sham_vectors = [
    make_sham_vector(
        5000 + i
    )
    for i in range(N_SHAMS)
]

print(
    "Created",
    N_SHAMS,
    "true sham vectors."
)

# ============================================================
# CHECK SHAM ↔ REAL EGO SIMILARITY
# ============================================================

print(
    "\nCosine similarity to real ego:"
)

for i, v in enumerate(
    sham_vectors
):

    cosine = torch.dot(
        ego_vector,
        v
    ).item()

    print(
        f"Sham {i+1}: {cosine:.4f}"
    )

# ============================================================
# Φ FUNCTION
# ============================================================

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

    return float(
        result.phi
    )

# ============================================================
# RUN REAL EGO + SHAMS
# ============================================================

results = []

conditions = [
    ("real_ego", ego_vector)
]

for i, vector in enumerate(
    sham_vectors
):

    conditions.append(
        (
            f"sham_{i+1}",
            vector
        )
    )


for name, vector in conditions:

    print(
        f"\n========== {name} =========="
    )

    for alpha in ALPHAS:

        phi = calculate_phi(
            vector,
            alpha
        )

        results.append({
            "condition": name,
            "alpha": alpha,
            "phi": phi
        })

        print(
            f"α={alpha:+3}  Φ={phi:.6f}"
        )

df_sham = pd.DataFrame(
    results
)

# ============================================================
# SHAM STATISTICS
# ============================================================

shams = df_sham[
    df_sham["condition"]
    .str.startswith("sham_")
]

sham_mean = (
    shams
    .groupby("alpha")["phi"]
    .mean()
    .reset_index()
)

sham_std = (
    shams
    .groupby("alpha")["phi"]
    .std()
    .reset_index()
    .rename(
        columns={"phi": "std"}
    )
)

real = df_sham[
    df_sham["condition"]
    == "real_ego"
]

comparison = (
    real[
        ["alpha", "phi"]
    ]
    .rename(
        columns={
            "phi": "ego_phi"
        }
    )
    .merge(
        sham_mean.rename(
            columns={
                "phi":
                "sham_mean_phi"
            }
        ),
        on="alpha"
    )
    .merge(
        sham_std,
        on="alpha"
    )
)

comparison[
    "ego_minus_sham"
] = (
    comparison["ego_phi"]
    - comparison["sham_mean_phi"]
)

comparison[
    "ego_sham_ratio"
] = (
    comparison["ego_phi"]
    / comparison["sham_mean_phi"]
)

# ============================================================
# PRINT
# ============================================================

print(
    "\n=============================="
)

print(
    "REAL EGO vs TRUE SHAM"
)

print(
    "=============================="
)

print(
    comparison.to_string(
        index=False
    )
)

# ============================================================
# PEAKS
# ============================================================

ego_peak = comparison.loc[
    comparison["ego_phi"].idxmax()
]

sham_peak = comparison.loc[
    comparison["sham_mean_phi"].idxmax()
]

print(
    "\n=============================="
)

print(
    "PEAK COMPARISON"
)

print(
    "=============================="
)

print(
    "Real ego:"
)

print(
    "α =",
    ego_peak["alpha"],
    "Φ =",
    ego_peak["ego_phi"]
)

print(
    "\nSham mean:"
)

print(
    "α =",
    sham_peak["alpha"],
    "Φ =",
    sham_peak["sham_mean_phi"]
)

# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(8,5)
)

# Individual sham curves
for name in shams[
    "condition"
].unique():

    subset = shams[
        shams["condition"] == name
    ]

    plt.plot(
        subset["alpha"],
        subset["phi"],
        marker="o",
        alpha=0.35
    )

# Sham mean
plt.plot(
    sham_mean["alpha"],
    sham_mean["phi"],
    marker="o",
    linewidth=3,
    label="true sham mean"
)

# Real ego
plt.plot(
    real["alpha"],
    real["phi"],
    marker="o",
    linewidth=3,
    label="real ego"
)

plt.xlabel(
    "Steering strength α"
)

plt.ylabel(
    "IIT Φ"
)

plt.title(
    "Real Ego vs True Sham Controls"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "real_ego_vs_true_sham.png",
    dpi=300
)

plt.show()

# ============================================================
# SAVE
# ============================================================

df_sham.to_csv(
    "real_ego_vs_true_sham.csv",
    index=False
)

comparison.to_csv(
    "real_ego_vs_true_sham_comparison.csv",
    index=False
)

print(
    "\nSaved true-sham results."
)
