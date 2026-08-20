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
N_EGOS=5
N_NULLS=30
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
# EGO DATA
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

# ============================================================
# PCA / 3-NODE SYSTEM
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

calibration=np.stack([get_activation(x).cpu().numpy() for x in calibration_texts])

pca=PCA(n_components=N_NODES)
pca.fit(calibration)

basis=torch.tensor(pca.components_,dtype=torch.float32,device=DEVICE)
baseline=torch.tensor(pca.mean_,dtype=torch.float32,device=DEVICE)

states=list(product([0,1],repeat=N_NODES))
N_STATES=len(states)

state_tensor=torch.tensor(states,dtype=torch.float32,device=DEVICE)
encoded_states=baseline.unsqueeze(0)+(2*state_tensor-1)@basis

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
                fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post",hook)]
            ):
                _,cache=model.run_with_cache(tokens)

        outputs.append(
            cache[f"blocks.{READ_LAYER}.hook_resid_post"][:,-1,:].detach()
        )

    return torch.cat(outputs,dim=0)

# ============================================================
# PYΦ
# ============================================================
cm=np.ones((N_NODES,N_NODES),dtype=int)

def phi_for_vector(vector,alpha):
    next_activation=propagate(vector,alpha)
    coords=(next_activation-baseline)@basis.T
    tpm=torch.sigmoid(coords).cpu().numpy()
    tpm=np.clip(tpm,1e-5,1-1e-5)

    substrate=pyphi.Substrate(tpm=tpm,cm=cm)

    return float(
        pyphi.analyze(
            substrate,
            TEST_STATE,
            formalism="IIT_3_0",
            compute="sia"
        ).phi
    )

# ============================================================
# U-SCORE
# ============================================================
def u_score(y):
    y=np.asarray(y)
    line=np.linspace(y[0],y[-1],len(y))
    return float(np.max(y-line))

# ============================================================
# EGO VECTORS
# ============================================================
def make_ego(seed):
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(high),len(high))

    v=high[idx].mean(0)-low[idx].mean(0)
    v/=v.norm()
    return v

ego_vectors=[make_ego(2000+i) for i in range(N_EGOS)]

# Include original vector
original_ego=high.mean(0)-low.mean(0)
original_ego/=original_ego.norm()
ego_vectors[0]=original_ego

# ============================================================
# EGO TRAJECTORIES
# ============================================================
ego_results=[]

for i,v in enumerate(ego_vectors):
    print(f"\n========== EGO {i+1}/{N_EGOS} ==========")

    y=[]

    for alpha in ALPHAS:
        phi=phi_for_vector(v,alpha)
        y.append(phi)
        print(f"α={alpha:+} Φ={phi:.6f}")

    score=u_score(y)

    ego_results.append({
        "ego_id":i+1,
        "u_score":score,
        "phi_values":y
    })

    print("U-score:",score)

# ============================================================
# RANDOM ORTHOGONAL NULLS
# ============================================================
def random_null(seed):
    torch.manual_seed(seed)
    v=torch.randn_like(original_ego)
    v-=torch.dot(v,original_ego)*original_ego
    v/=v.norm()
    return v*original_ego.norm()

null_scores=[]

for i in range(N_NULLS):
    print(f"\n========== NULL {i+1}/{N_NULLS} ==========")

    v=random_null(50000+i)
    y=[]

    for alpha in ALPHAS:
        phi=phi_for_vector(v,alpha)
        y.append(phi)
        print(f"α={alpha:+} Φ={phi:.6f}")

    score=u_score(y)
    null_scores.append(score)

    print("U-score:",score)

# ============================================================
# EGO SUMMARY
# ============================================================
ego_scores=np.array([
    x["u_score"] for x in ego_results
])

null_scores=np.array(null_scores)

ego_mean=ego_scores.mean()
ego_sd=ego_scores.std()

null_mean=null_scores.mean()
null_sd=null_scores.std()

# Compare mean ego U-score against null distribution
p=np.mean(
    null_scores>=ego_mean
)

z=(
    ego_mean-null_mean
)/null_sd

print("\n==============================")
print("FINAL ROBUSTNESS TEST")
print("==============================")

print("Ego U-scores:",ego_scores)
print("Ego mean:",ego_mean)
print("Ego SD:",ego_sd)
print("Null mean:",null_mean)
print("Null SD:",null_sd)
print("Null max:",null_scores.max())
print("Empirical p:",(np.sum(null_scores>=ego_mean)+1)/(N_NULLS+1))
print("Ego mean z-score:",z)

# ============================================================
# SAVE
# ============================================================
pd.DataFrame({
    "ego_id":np.arange(1,N_EGOS+1),
    "u_score":ego_scores
}).to_csv("ego_vector_robustness.csv",index=False)

pd.DataFrame({
    "null_id":np.arange(1,N_NULLS+1),
    "u_score":null_scores
}).to_csv("null_robustness.csv",index=False)

# ============================================================
# PLOT U-SCORES
# ============================================================
plt.figure(figsize=(8,5))

plt.scatter(
    np.arange(len(null_scores)),
    null_scores,
    alpha=0.5,
    label="Random nulls"
)

plt.axhline(
    ego_mean,
    linestyle="--",
    linewidth=3,
    label="Mean ego U-score"
)

plt.xlabel("Random null index")
plt.ylabel("Inverted-U score")
plt.title("Ego robustness vs random null distribution")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# PLOT EGO TRAJECTORIES
# ============================================================
plt.figure(figsize=(8,5))

for x in ego_results:
    plt.plot(
        ALPHAS,
        x["phi_values"],
        marker="o",
        alpha=0.7
    )

plt.xlabel("Steering strength α")
plt.ylabel("IIT Φ")
plt.title("Independent ego-vector trajectories")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %%
print("\n==============================")
print("FINAL ROBUSTNESS TEST")
print("==============================")

print("Ego U-scores:",ego_scores)
print("Ego mean:",ego_mean)
print("Ego SD:",ego_sd)
print("Null mean:",null_mean)
print("Null SD:",null_sd)
print("Null max:",null_scores.max())
print("Empirical p:",(np.sum(null_scores>=ego_mean)+1)/(N_NULLS+1))
print("Ego mean z-score:",z)

# ============================================================
# SAVE
# ============================================================
pd.DataFrame({
    "ego_id":np.arange(1,N_EGOS+1),
    "u_score":ego_scores
}).to_csv("ego_vector_robustness.csv",index=False)

pd.DataFrame({
    "null_id":np.arange(1,N_NULLS+1),
    "u_score":null_scores
}).to_csv("null_robustness.csv",index=False)

# ============================================================
# PLOT U-SCORES
# ============================================================
plt.figure(figsize=(8,5))

plt.scatter(
    np.arange(len(null_scores)),
    null_scores,
    alpha=0.5,
    label="Random nulls"
)

plt.axhline(
    ego_mean,
    linestyle="--",
    linewidth=3,
    label="Mean ego U-score"
)

plt.xlabel("Random null index")
plt.ylabel("Inverted-U score")
plt.title("Ego robustness vs random null distribution")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# PLOT EGO TRAJECTORIES
# ============================================================
plt.figure(figsize=(8,5))

for x in ego_results:
    plt.plot(
        ALPHAS,
        x["phi_values"],
        marker="o",
        alpha=0.7
    )

plt.xlabel("Steering strength α")
plt.ylabel("IIT Φ")
plt.title("Independent ego-vector trajectories")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
