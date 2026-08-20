# %%
# ============================================================
# QUANTIFY EGO vs CONTROLS
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import permutation_test

# ------------------------------------------------------------
# Separate ego and controls
# ------------------------------------------------------------

ego = (
    df[df["condition"] == "ego"]
    .sort_values("alpha")
    .reset_index(drop=True)
)

controls = df[
    df["condition"].str.startswith("control")
].copy()

control_mean = (
    controls
    .groupby("alpha")["phi"]
    .mean()
    .reset_index()
)

control_std = (
    controls
    .groupby("alpha")["phi"]
    .std()
    .reset_index()
)

# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

comparison = ego[
    ["alpha", "phi"]
].rename(
    columns={"phi": "ego_phi"}
).merge(
    control_mean.rename(
        columns={"phi": "control_mean_phi"}
    ),
    on="alpha"
).merge(
    control_std.rename(
        columns={"phi": "control_std_phi"}
    ),
    on="alpha"
)

comparison["ego_minus_control"] = (
    comparison["ego_phi"]
    - comparison["control_mean_phi"]
)

comparison["ego_ratio"] = (
    comparison["ego_phi"]
    / comparison["control_mean_phi"]
)

print("\n==============================")
print("EGO vs CONTROL")
print("==============================")

print(
    comparison.to_string(index=False)
)

# ============================================================
# PEAK
# ============================================================

ego_peak = ego.loc[
    ego["phi"].idxmax()
]

control_peak = control_mean.loc[
    control_mean["phi"].idxmax()
]

print("\n==============================")
print("PEAK COMPARISON")
print("==============================")

print(
    "Ego peak Φ:",
    ego_peak["phi"]
)

print(
    "Ego peak α:",
    ego_peak["alpha"]
)

print(
    "Control maximum mean Φ:",
    control_peak["phi"]
)

print(
    "Control maximum α:",
    control_peak["alpha"]
)

print(
    "Peak excess:",
    ego_peak["phi"]
    - control_peak["phi"]
)

# ============================================================
# AREA UNDER CURVE
# ============================================================

ego_auc = np.trapezoid(
    ego["phi"],
    ego["alpha"]
)

control_auc = np.trapezoid(
    control_mean["phi"],
    control_mean["alpha"]
)

print("\n==============================")
print("AREA UNDER CURVE")
print("==============================")

print(
    "Ego AUC:",
    ego_auc
)

print(
    "Control AUC:",
    control_auc
)

print(
    "AUC difference:",
    ego_auc - control_auc
)

print(
    "AUC ratio:",
    ego_auc / control_auc
)

# ============================================================
# TURNING POINT
# ============================================================

peak_index = ego["phi"].idxmax()

if (
    peak_index > 0
    and peak_index < len(ego) - 1
):

    left = ego.iloc[
        peak_index - 1
    ]

    peak = ego.iloc[
        peak_index
    ]

    right = ego.iloc[
        peak_index + 1
    ]

    rising = (
        peak["phi"] - left["phi"]
    )

    falling = (
        peak["phi"] - right["phi"]
    )

    print("\n==============================")
    print("INVERTED-U CHECK")
    print("==============================")

    print(
        "Rise into peak:",
        rising
    )

    print(
        "Fall after peak:",
        falling
    )

    if rising > 0 and falling > 0:
        print(
            "✓ Local inverted-U detected."
        )
    else:
        print(
            "✗ No local inverted-U."
        )

# ============================================================
# POINTWISE EFFECT
# ============================================================

print("\n==============================")
print("LARGEST EGO EFFECTS")
print("==============================")

print(
    comparison
    .sort_values(
        "ego_minus_control",
        ascending=False
    )
    [["alpha",
      "ego_phi",
      "control_mean_phi",
      "ego_minus_control"]]
    .to_string(index=False)
)

# ============================================================
# SAVE
# ============================================================

comparison.to_csv(
    "ego_control_quantification.csv",
    index=False
)

print(
    "\nSaved: ego_control_quantification.csv"
)
