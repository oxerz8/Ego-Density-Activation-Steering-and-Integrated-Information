# %%
# ============================================================
# RANDOM NULL-DIRECTION CONTROL
# ============================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pyphi

ALPHAS = [-10,-6,-2,0,2,6,10,14,18,20]
N_CONTROLS = 10
TEST_STATE = (1,0,1)

cm = np.ones(
    (N_NODES, N_NODES),
    dtype=int
)

# ============================================================
# RANDOM DIRECTIONS
# ============================================================

def make_random_null(seed):

    torch.manual_seed(seed)

    v = torch.randn_like(
        ego_vector
    )

    # Remove any component parallel to ego.
    v = (
        v
        - torch.dot(v, ego_vector)
        * ego_vector
    )

    # Match ego-vector norm exactly.
    v = (
        v / v.norm()
        * ego_vector.norm()
    )

    return v


random_controls = [
    make_random_null(10000 + i)
    for i in range(N_CONTROLS)
]

print("\nRANDOM CONTROL CHECK")

for i, v in enumerate(random_controls):

    cosine = torch.dot(
        v / v.norm(),
        ego_vector / ego_vector.norm()
    ).item()

    print(
        f"Control {i+1}: cosine = {cosine:.6f}"
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

    return float(result.phi)

# ============================================================
# RUN EGO + RANDOM CONTROLS
# ============================================================

results = []

conditions = [
    ("ego", ego_vector)
]

for i, v in enumerate(random_controls):
    conditions.append(
        (f"random_{i+1}", v)
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

df_random = pd.DataFrame(
    results
)

# ============================================================
# INVERTED-U SCORE
#
# Positive score means the curve rises above the straight
# line connecting its two endpoints.
# ============================================================

def inverted_u_score(group):

    group = group.sort_values(
        "alpha"
    )

    x = group["alpha"].to_numpy()
    y = group["phi"].to_numpy()

    # Straight line between endpoints
    baseline_line = np.linspace(
        y[0],
        y[-1],
        len(y)
    )

    residual = (
        y - baseline_line
    )

    return np.max(residual)


scores = []

for condition in df_random[
    "condition"
].unique():

    subset = df_random[
        df_random["condition"] == condition
    ]

    score = inverted_u_score(
        subset
    )

    scores.append({
        "condition": condition,
        "inverted_u_score": score
    })

scores_df = pd.DataFrame(
    scores
)

# ============================================================
# EGO VS RANDOM DISTRIBUTION
# ============================================================

ego_score = scores_df.loc[
    scores_df["condition"] == "ego",
    "inverted_u_score"
].iloc[0]

random_scores = scores_df[
    scores_df["condition"] != "ego"
]["inverted_u_score"]

percentile = (
    np.mean(
        random_scores < ego_score
    ) * 100
)

print(
    "\n=============================="
)

print(
    "INVERTED-U TEST"
)

print(
    "=============================="
)

print(
    "Ego inverted-U score:",
    ego_score
)

print(
    "Random mean:",
    random_scores.mean()
)

print(
    "Random SD:",
    random_scores.std()
)

print(
    "Ego percentile among random controls:",
    percentile,
    "%"
)

# ============================================================
# TABLE
# ============================================================

print(
    "\n=============================="
)

print(
    scores_df.sort_values(
        "inverted_u_score",
        ascending=False
    ).to_string(index=False)
)

# ============================================================
# PLOT ALL RANDOM CONTROLS
# ============================================================

plt.figure(
    figsize=(9,6)
)

for condition in df_random[
    "condition"
].unique():

    subset = df_random[
        df_random["condition"] == condition
    ]

    if condition == "ego":

        plt.plot(
            subset["alpha"],
            subset["phi"],
            marker="o",
            linewidth=4,
            label="EGO"
        )

    else:

        plt.plot(
            subset["alpha"],
            subset["phi"],
            marker="o",
            alpha=0.25
        )

plt.xlabel(
    "Steering strength α"
)

plt.ylabel(
    "IIT Φ"
)

plt.title(
    "Ego vs Random Orthogonal Steering Directions"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "ego_vs_random_nulls.png",
    dpi=300
)

plt.show()

# ============================================================
# SAVE
# ============================================================

df_random.to_csv(
    "ego_vs_random_nulls.csv",
    index=False
)

scores_df.to_csv(
    "ego_vs_random_null_scores.csv",
    index=False
)

print(
    "\nSaved random-null results."
)
