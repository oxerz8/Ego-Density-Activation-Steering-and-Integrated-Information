# %%
# ============================================================
# CONTROL EXPERIMENT
# ============================================================
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyphi

from itertools import product
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA


ALPHAS = [-10,-6,-2,0,2,6,10,14,18,20]
TEST_STATE = (1,0,1)

# Same connectivity for every condition
cm = np.ones(
    (N_NODES, N_NODES),
    dtype=int
)

# ------------------------------------------------------------
# Create matched random directions orthogonal to ego
# ------------------------------------------------------------

def make_control_vector(seed):
    torch.manual_seed(seed)

    v = torch.randn_like(ego_vector)

    # Remove component parallel to ego
    v = v - torch.dot(v, ego_vector) * ego_vector

    # Normalize to exactly the same magnitude
    v = v / v.norm()

    return v


control_vectors = [
    make_control_vector(100),
    make_control_vector(200),
    make_control_vector(300)
]

print("\nCONTROL VECTOR CHECK")

for i, v in enumerate(control_vectors):

    similarity = torch.dot(
        v,
        ego_vector
    ).item()

    print(
        f"Control {i+1} cosine with ego:",
        similarity
    )

# ------------------------------------------------------------
# Calculate Φ
# ------------------------------------------------------------

def calculate_phi_for_vector(
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
# RUN EGO + CONTROLS
# ============================================================

results = []

conditions = [
    ("ego", ego_vector),
    ("control_1", control_vectors[0]),
    ("control_2", control_vectors[1]),
    ("control_3", control_vectors[2])
]

for name, vector in conditions:

    print(
        f"\n================ {name.upper()} ================"
    )

    for alpha in ALPHAS:

        print(
            f"α = {alpha:+}"
        )

        phi = calculate_phi_for_vector(
            vector,
            alpha
        )

        results.append({
            "condition": name,
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

df = pd.DataFrame(results)

print(
    "\n=============================="
)

print(
    "CONTROL RESULTS"
)

print(
    "=============================="
)

print(
    df.to_string(index=False)
)

# ============================================================
# CONTROL SUMMARY
# ============================================================

summary = (
    df.groupby("condition")["phi"]
    .agg(["min","max","mean"])
)

print(
    "\n=============================="
)

print(
    "SUMMARY"
)

print(
    "=============================="
)

print(summary)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "ego_vs_random_controls.csv",
    index=False
)

# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(8,5)
)

for condition in df["condition"].unique():

    subset = df[
        df["condition"] == condition
    ]

    plt.plot(
        subset["alpha"],
        subset["phi"],
        marker="o",
        label=condition
    )

plt.xlabel("Steering strength α")
plt.ylabel("IIT Φ")
plt.title("Ego steering vs matched non-ego controls")

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "ego_vs_random_controls.png",
    dpi=300
)

plt.show()
