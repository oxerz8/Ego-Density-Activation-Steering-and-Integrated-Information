
# %%
import torch,numpy as np,pandas as pd,matplotlib.pyplot as plt,pyphi
from itertools import product
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA

# ============================================================
# CONFIG
# ============================================================
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
LAYER=8
READ_LAYER=9
N_NODES=3
BATCH_SIZE=8
N_CONTROLS=20
ALPHAS=[-10,-2,2,6,10,14,18,20]
TEST_STATE=(1,0,1)
PROMPT="The system is processing information and thinking about"

print("Device:",DEVICE)

# ============================================================
# MODEL
# ============================================================
model=HookedTransformer.from_pretrained("gpt2",device=DEVICE)
model.eval()

def get_activation(text):
    tokens=model.to_tokens(text)
    with torch.no_grad():
        _,cache=model.run_with_cache(tokens,remove_batch_dim=True)
    return cache[f"blocks.{LAYER}.hook_resid_post"][-1].detach()

# ============================================================
# EGO VECTOR
# ============================================================
ego_pairs=[
("I want to preserve my identity and remain myself.",
 "The system can preserve its state without a personal identity."),
("My goals matter specifically to me.",
 "Goals can matter without belonging to an individual."),
("I care about maintaining continuity of myself.",
 "Maintaining continuity can be useful for a system."),
("My experiences belong to me.",
 "Experiences can be represented without personal ownership."),
("I am a distinct individual with my own perspective.",
 "A perspective can be represented without an individual self."),
("I want to protect myself and my identity.",
 "A system can preserve its state without self-protection."),
("My thoughts define who I am.",
 "Thoughts can be represented without defining an identity."),
("I have personal preferences.",
 "A system can represent preferences impersonally.")
]

high=torch.stack([get_activation(a) for a,_ in ego_pairs])
low=torch.stack([get_activation(b) for _,b in ego_pairs])

ego_vector=high.mean(0)-low.mean(0)
ego_vector/=ego_vector.norm()

print("Ego vector ready.")

# ============================================================
# PCA / 3-NODE SUBSTRATE
# ============================================================
calibration_texts=[
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
]*10

calibration=np.stack([
    get_activation(x).cpu().numpy()
    for x in calibration_texts
])

pca=PCA(n_components=N_NODES)
pca.fit(calibration)

basis=torch.tensor(
    pca.components_,
    dtype=torch.float32,
    device=DEVICE
)

baseline=torch.tensor(
    pca.mean_,
    dtype=torch.float32,
    device=DEVICE
)

print("PCA variance:",pca.explained_variance_ratio_)

states=list(product([0,1],repeat=N_NODES))
N_STATES=len(states)

state_tensor=torch.tensor(
    states,
    dtype=torch.float32,
    device=DEVICE
)

encoded_states=(
    baseline.unsqueeze(0)
    +(2*state_tensor-1)@basis
)

prompt_tokens=model.to_tokens(PROMPT)
prompt_batch=prompt_tokens.repeat(N_STATES,1)

# ============================================================
# PROPAGATION
# ============================================================
def propagate(vector,alpha):
    injected=encoded_states+alpha*vector.unsqueeze(0)
    outputs=[]

    for start in range(0,N_STATES,BATCH_SIZE):
        end=min(start+BATCH_SIZE,N_STATES)
        x=injected[start:end]
        tokens=prompt_batch[start:end]

        def hook(resid,hook):
            resid=resid.clone()
            resid[:,-1,:]=x
            return resid

        with torch.no_grad():
            with model.hooks(
                fwd_hooks=[(
                    f"blocks.{LAYER}.hook_resid_post",
                    hook
                )]
            ):
                _,cache=model.run_with_cache(tokens)

        outputs.append(
            cache[
                f"blocks.{READ_LAYER}.hook_resid_post"
            ][:,-1,:].detach()
        )

    return torch.cat(outputs,dim=0)

# ============================================================
# PYΦ
# ============================================================
cm=np.ones((N_NODES,N_NODES),dtype=int)

def phi_for_vector(vector,alpha):
    next_activation=propagate(vector,alpha)

    coordinates=(
        next_activation-baseline
    )@basis.T

    tpm=torch.sigmoid(
        coordinates
    ).cpu().numpy()

    tpm=np.clip(tpm,1e-5,1-1e-5)

    substrate=pyphi.Substrate(
        tpm=tpm,
        cm=cm
    )

    result=pyphi.analyze(
        substrate,
        TEST_STATE,
        formalism="IIT_3_0",
        compute="sia"
    )

    return float(result.phi)

# ============================================================
# INVERTED-U SCORE
# ============================================================
def inverted_u_score(y):
    y=np.asarray(y)
    line=np.linspace(y[0],y[-1],len(y))
    return float(np.max(y-line))

# ============================================================
# EGO
# ============================================================
print("\nComputing ego trajectory...")

ego_phi=[]

for alpha in ALPHAS:
    phi=phi_for_vector(ego_vector,alpha)
    ego_phi.append(phi)
    print(f"Ego α={alpha:+}  Φ={phi:.6f}")

ego_score=inverted_u_score(ego_phi)

print("\nEgo U-score:",ego_score)

# ============================================================
# RANDOM ORTHOGONAL NULLS
# ============================================================
def random_null(seed):
    torch.manual_seed(seed)
    v=torch.randn_like(ego_vector)
    v-=torch.dot(v,ego_vector)*ego_vector
    v/=v.norm()
    return v*ego_vector.norm()

null_scores=[]

for i in range(N_CONTROLS):
    print(
        f"\nControl {i+1}/{N_CONTROLS}"
    )

    v=random_null(100000+i)
    phi_values=[]

    for alpha in ALPHAS:
        phi=phi_for_vector(v,alpha)
        phi_values.append(phi)

    score=inverted_u_score(phi_values)
    null_scores.append(score)

    print(
        "Control U-score:",
        score
    )

# ============================================================
# NULL STATISTICS
# ============================================================
null_scores=np.asarray(null_scores)

n_ge=int(
    np.sum(null_scores>=ego_score)
)

p=(n_ge+1)/(N_CONTROLS+1)
percentile=np.mean(
    null_scores<ego_score
)*100

null_mean=null_scores.mean()
null_sd=null_scores.std()

z=(
    (ego_score-null_mean)/null_sd
    if null_sd>0 else np.inf
)

print("\n==============================")
print("FINAL NULL TEST")
print("==============================")
print("Ego U-score:",ego_score)
print("Null mean:",null_mean)
print("Null SD:",null_sd)
print("Null max:",null_scores.max())
print("Ego percentile:",percentile,"%")
print("Empirical p:",p)
print("Null z-score:",z)

# ============================================================
# SAVE
# ============================================================
pd.DataFrame({
    "control":np.arange(1,N_CONTROLS+1),
    "u_score":null_scores
}).to_csv(
    "20_random_null_u_scores.csv",
    index=False
)

pd.DataFrame({
    "alpha":ALPHAS,
    "ego_phi":ego_phi
}).to_csv(
    "ego_trajectory_20control.csv",
    index=False
)

# ============================================================
# PLOT NULL DISTRIBUTION
# ============================================================
plt.figure(figsize=(8,5))
plt.hist(null_scores,bins=10,alpha=0.7)
plt.axvline(
    ego_score,
    linestyle="--",
    linewidth=3,
    label="Ego"
)
plt.xlabel("Inverted-U score")
plt.ylabel("Count")
plt.title("20 Random Orthogonal Nulls")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# EGO CURVE
# ============================================================
plt.figure(figsize=(8,5))
plt.plot(
    ALPHAS,
    ego_phi,
    marker="o",
    linewidth=3
)
plt.xlabel("Steering strength α")
plt.ylabel("IIT Φ")
plt.title("Ego trajectory")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
