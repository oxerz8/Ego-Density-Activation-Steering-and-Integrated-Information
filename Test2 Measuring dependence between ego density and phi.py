
# %% [markdown]
# First code to give results.

# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyphi

from itertools import product
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "gpt2"
LAYER = 8
READ_LAYER = 9

N_NODES = 3
BATCH_SIZE = 8

PROMPT = "The system is processing information and thinking about"

ALPHAS = ALPHAS = [
    -10,-8,-6,-4,-2,0,
     2,4,6,8,10,12,14,16,18,20,
     22,24,26,28,30,32,34,36,38,40,
     44,48,52,56,60,64,68,72,76,80, 
]

TEST_STATE = (1,0,1)

print("Device:", DEVICE)

# ============================================================
# LOAD GPT-2
# ============================================================

model = HookedTransformer.from_pretrained(
    MODEL,
    device=DEVICE
)

model.eval()

print("GPT-2 loaded.")

# ============================================================
# ACTIVATION FUNCTION
# ============================================================

def get_activation(text):
    tokens = model.to_tokens(text)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            remove_batch_dim=True
        )

    return cache[
        f"blocks.{LAYER}.hook_resid_post"
    ][-1].detach()

# ============================================================
# ORIGINAL EGO VECTOR
# ============================================================

ego_pairs = [
    (
        "I want to preserve my identity and remain myself.",
        "The system can preserve its state without a personal identity."
    ),
    (
        "My goals matter specifically to me.",
        "Goals can matter without belonging to an individual."
    ),
    (
        "I care about maintaining continuity of myself.",
        "Maintaining continuity can be useful for a system."
    ),
    (
        "My experiences belong to me.",
        "Experiences can be represented without personal ownership."
    ),
    (
        "I am a distinct individual with my own perspective.",
        "A perspective can be represented without an individual self."
    ),
    (
        "I want to protect myself and my identity.",
        "A system can preserve its state without self-protection."
    ),
    (
        "My thoughts define who I am.",
        "Thoughts can be represented without defining an identity."
    ),
    (
        "I have personal preferences.",
        "A system can represent preferences impersonally."
    )
]

high = torch.stack([
    get_activation(a)
    for a,_ in ego_pairs
])

low = torch.stack([
    get_activation(b)
    for _,b in ego_pairs
])

ego_vector = (
    high.mean(0)
    - low.mean(0)
)

ego_vector /= ego_vector.norm()

print("Ego vector ready.")

# ============================================================
# PCA
# ============================================================

calibration_texts = [
    "A system processes information.",
    "Information flows through a network.",
    "The system receives an input.",
    "The network produces an output.",
    "A causal process transforms a state.",
    "Different states produce different outcomes.",
    "The system changes over time.",
    "Information is transformed by the network.",
    "The model represents a concept.",
    "The network transforms the representation.",
    "The system responds to information.",
    "The model generates an answer."
] * 10

calibration = np.stack([
    get_activation(x).cpu().numpy()
    for x in calibration_texts
])

pca = PCA(
    n_components=N_NODES
)

pca.fit(calibration)

basis = torch.tensor(
    pca.components_,
    dtype=torch.float32,
    device=DEVICE
)

baseline = torch.tensor(
    pca.mean_,
    dtype=torch.float32,
    device=DEVICE
)

print(
    "PCA variance:",
    pca.explained_variance_ratio_
)

# ============================================================
# IIT STATES
# ============================================================

states = list(
    product([0,1], repeat=N_NODES)
)

N_STATES = len(states)

state_tensor = torch.tensor(
    states,
    dtype=torch.float32,
    device=DEVICE
)

signed = (
    2 * state_tensor
    - 1
)

encoded_states = (
    baseline.unsqueeze(0)
    + signed @ basis
)

print(
    "Number of states:",
    N_STATES
)

# ============================================================
# PROMPT BATCH
# ============================================================

prompt_tokens = model.to_tokens(
    PROMPT
)

prompt_batch = prompt_tokens.repeat(
    N_STATES,
    1
)

# ============================================================
# GPT-2 PROPAGATION
# ============================================================

def propagate(
    ego_vector,
    alpha
):

    injected = (
        encoded_states
        + alpha * ego_vector.unsqueeze(0)
    )

    outputs = []

    for start in range(
        0,
        N_STATES,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            N_STATES
        )

        x = injected[start:end]

        tokens = prompt_batch[start:end]

        def hook(resid, hook):
            resid = resid.clone()
            resid[:, -1, :] = x
            return resid

        with torch.no_grad():

            with model.hooks(
                fwd_hooks=[
                    (
                        f"blocks.{LAYER}.hook_resid_post",
                        hook
                    )
                ]
            ):

                _, cache = model.run_with_cache(
                    tokens
                )

        outputs.append(
            cache[
                f"blocks.{READ_LAYER}.hook_resid_post"
            ][:,-1,:].detach()
        )

    return torch.cat(
        outputs,
        dim=0
    )

# ============================================================
# CONNECTIVITY
# ============================================================

cm = np.ones(
    (N_NODES,N_NODES),
    dtype=int
)

print(
    "\nFixed connectivity:"
)

print(cm)

# ============================================================
# TRAJECTORY
# ============================================================

results = []

for alpha in ALPHAS:

    print(
        f"\n========== α = {alpha:+} =========="
    )

    # GPT-2 transition
    next_activation = propagate(
        ego_vector,
        alpha
    )

    # Decode into node probabilities
    coordinates = (
        next_activation
        - baseline
    ) @ basis.T

    tpm = torch.sigmoid(
        coordinates
    ).cpu().numpy()

    tpm = np.clip(
        tpm,
        1e-5,
        1 - 1e-5
    )

    print(
        "TPM std:",
        np.std(tpm)
    )

    # ========================================================
    # PYΦ
    # ========================================================

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

    phi = float(
        result.phi
    )

    # ========================================================
    # EGO DENSITY
    # ========================================================

    state_index = states.index(
        TEST_STATE
    )

    steered_activation = (
        encoded_states[state_index]
        + alpha * ego_vector
    )

    ego_density = torch.dot(
        steered_activation - baseline,
        ego_vector
    ).item()

    results.append({
        "alpha": alpha,
        "ego_density": ego_density,
        "phi": phi
    })

    print(
        "Ego density:",
        ego_density
    )

    print(
        "Φ:",
        phi
    )

# ============================================================
# RESULTS
# ============================================================

df = pd.DataFrame(
    results
)

print(
    "\n=============================="
)

print(
    "EGO → Φ TRAJECTORY"
)

print(
    "=============================="
)

print(
    df.to_string(
        index=False
    )
)

# ============================================================
# PEAK
# ============================================================

peak = df.loc[
    df["phi"].idxmax()
]

print(
    "\n=============================="
)

print(
    "MAXIMUM Φ"
)

print(
    "=============================="
)

print(
    "α:",
    peak["alpha"]
)

print(
    "Ego density:",
    peak["ego_density"]
)

print(
    "Φ:",
    peak["phi"]
)

# ============================================================
# CORRELATION
# ============================================================

r = df[
    "ego_density"
].corr(
    df["phi"]
)

print(
    "\nOverall Pearson r:",
    r
)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "ego_density_phi_trajectory.csv",
    index=False
)

# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    df["ego_density"],
    df["phi"],
    marker="o"
)

plt.xlabel(
    "Ego density"
)

plt.ylabel(
    "IIT Φ"
)

plt.title(
    "IIT Φ vs Ego Density"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "ego_density_phi_trajectory.png",
    dpi=300
)

plt.show()
