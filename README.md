# Ego Density, Activation Steering, and Integrated Information

A computational investigation of the relationship between **self-referential representations, ego density, and IIT integrated information (Φ)** in GPT-2.

> **Status:** Preliminary research / experimental prototype
> **Model:** GPT-2 Small
> **Frameworks:** Mechanistic Interpretability, Activation Steering, Integrated Information Theory (IIT 3.0), PyPhi
> **Primary hypothesis:** Ego-like organization may initially increase integrated information, but excessive ego density may become counterproductive, producing an inverted-U relationship between ego density and Φ.

---

## Overview

This project investigates a computational hypothesis emerging from the **Awareness–Thought Space / Dual Aspect Monism (DAM)** framework.

The underlying theoretical framework proposes a psychophysically neutral Awareness Space containing an awareness field and localized concentration centers interpreted as observers. Thoughts are represented as vectors in a Thought Space, and interactions between an observer and thoughts produce stabilization/mental inertia and an Ego structure. The manuscript formalizes Ego density as accumulated thought intensity divided by the volume of the self-awareness region.   

The computational project asks whether an analogous phenomenon can be observed in the internal dynamics of a language model.

The central empirical question is:

[
$\boxed{\text{How does IIT }\Phi\text{ change as ego-like representation increases?}}$
]

---

# 1. Initial Representation-Space Finding

The project began with the observation that self-referential thoughts might form a geometrically concentrated region in transformer representation space.

Using GPT-2 Small (12 layers, 768-dimensional residual representations), self-referential and objective sentences were compared using final-token residual-stream activations.

### Cosine similarity

| Representation group  | Mean cosine similarity |
| --------------------- | ---------------------: |
| Self → Self           |             **0.9794** |
| Objective → Objective |             **0.9510** |
| Self → Objective      |             **0.9475** |

Self-referential representations were therefore substantially more internally cohesive than the objective group.

### PCA

A 2D PCA projection produced visible separation between the two groups.

PC1 explained approximately **24.1% of the variance**.
<img width="2400" height="1800" alt="pca_thought_space_clustering" src="https://github.com/user-attachments/assets/8dfb2fcc-eab3-4da7-b4e2-c7d3cb8abb4e" />

---

# 2. Layer-by-Layer Emergence

The self/objective separation was then measured throughout GPT-2.

The observed pattern was:

* **Layers 0–2:** little differentiation.
* **Layers 3–8:** increasing separation.
* **Layer 8:** strongest observed separation.
* **Layers 9–11:** representations became more compressed again.

At Layer 8, cross-group similarity reached approximately **0.800**, while self-referential representations remained comparatively cohesive.

This made Layer 8 the primary location for the later causal experiments.
<img width="3000" height="1800" alt="layer_evolution" src="https://github.com/user-attachments/assets/db425d4b-3171-458e-a40f-35f00bd0fbf3" />

---

# 3. Layer-8 Circuit Analysis

Attention heads in Layer 8 were also inspected.

Reported observations included:

* **Heads 8.9 and 8.3:** preferential attention toward self-referential subject tokens.
* **Head 8.7:** broad attention-sink behavior.

Layer 8 was therefore selected **before the IIT experiments**, based on the earlier representation-space investigation rather than because it happened to produce an interesting Φ result.
<img width="3000" height="1500" alt="layer_8_heads" src="https://github.com/user-attachments/assets/4bda531f-077a-45aa-a85b-34cc861e5040" />

---

# 4. Activation Steering

The self-referential representation was converted into an activation-steering vector:

[
$v_{\mathrm{ego}}$
================

$## \operatorname{mean}(H_{\mathrm{self}})$

$\operatorname{mean}(H_{\mathrm{objective}})$
]

The residual stream was then intervened on according to:

[
$h'=h+\alpha v_{\mathrm{ego}}$
]

where:

$* (v_{\mathrm{ego}}) = ego/self-referential direction$
$* (\alpha) = steering strength$

This provides a controllable experimental proxy for **ego density**.

---

# 5. Early Causal-Integration Attempt

An initial experiment used a causal-disruption metric as a proxy for integration.

Although it produced strong relationships with steering strength, it was **not IIT Φ**.

That metric was therefore discarded as evidence for the main hypothesis.

The experiment was rebuilt using actual IIT-3 calculations with PyPhi.

---

# 6. PyPhi / IIT Validation

PyPhi was configured for **IIT 3.0**.

A built-in PyPhi sanity test successfully produced:

[
$\boxed{\Phi=2.312499}$
]

This established that the IIT implementation and environment were functioning correctly.

---

# 7. GPT-2 → IIT Causal Substrate

The next challenge was converting GPT-2 internal dynamics into a computational system small enough for exact IIT analysis.

Large systems were computationally prohibitive, so the project was reduced to a **3-node causal substrate**.

A binary system with 3 nodes has:

[
$2^3=8$
]

possible states.

Candidate GPT-2 activation directions were perturbed to measure causal interactions between internal features.

A larger 8-node causal interaction experiment found:

[
$\boxed{35\text{ directed causal edges}}$
]

among the candidate nodes, demonstrating that GPT-2 internal representations could be used to construct a nontrivial causal system.

---

# 8. Working GPT-2 → TPM → IIT Pipeline

A GPT-2-derived stochastic transition probability matrix (TPM) was constructed from the 3-node representation.

The resulting TPM showed:

* substantial variation;
* all 8 states represented distinctly;
* measurable cross-node dependencies.

The resulting system could be successfully passed to PyPhi.

At zero ego steering:

[
$\boxed{\Phi\approx0.001587}$
]

This established the working experimental pipeline:

```text
GPT-2 activation
      ↓
3-node causal representation
      ↓
TPM
      ↓
PyPhi / IIT 3.0
      ↓
Φ
```

---

# 9. Initial Ego-Density Sweep

The original ego vector was steered over:

[
$\alpha=-2,-1,0,+1,+2$
]

Observed results:

|  α | Ego density |        Φ |
| -: | ----------: | -------: |
| −2 |     −2.2296 | 0.001432 |
| −1 |     −1.2296 | 0.001390 |
|  0 |     −0.2296 | 0.001587 |
| +1 |      0.7704 | 0.002292 |
| +2 |      1.7704 | 0.003047 |

Pearson correlation:

[
$r\approx0.916$
]

Thus, in the initial regime:

[
$\boxed{E\uparrow\Rightarrow\Phi\uparrow}$
]

This contradicted the project's original assumption that increasing ego would immediately reduce awareness/integration.

---

# 10. Revised Hypothesis: Inverted-U

The hypothesis was consequently revised.

Instead of:

[
$E\uparrow\Rightarrow\Phi\downarrow$
]

the working hypothesis became:

[$
\boxed{
E\uparrow
\Rightarrow
\Phi\uparrow
\quad\text{initially}
}$
]

followed by:

[$
\boxed{
E>E^*
\Rightarrow
\Phi\downarrow
}$
]

The prediction is therefore an **inverted-U relationship**:

```text
Φ
│          /\
│         /  \
│        /    \
│_______/      \_______
└──────────────────────→ Ego density
```

The proposed interpretation is that a moderate degree of self-model/ego organization can improve integration, organization, agency, and adaptive coordination, while excessive self-referential organization may eventually become counterproductive.

---

# 11. Extended Ego-Density Trajectory

The steering range was expanded substantially.

The resulting trajectory exhibited:

1. an initial increase in Φ;
2. a peak at moderate/high ego density;
3. a subsequent decrease;
4. eventual collapse toward approximately zero Φ at extreme steering.

The peak occurred around the vicinity of:

[
$\alpha\approx8-10$
]

in the tested setup.

This was the first observation matching the revised inverted-U hypothesis.
<img width="2400" height="1500" alt="ego_density_phi_trajectory" src="https://github.com/user-attachments/assets/cbd67ca2-24ba-4b98-89f5-6040b22d9306" />

---

# 12. Control Experiments

Several controls were explored.

### Random orthogonal steering

Random directions were generated orthogonal to the ego vector and normalized to the same magnitude.

The ego trajectory was substantially stronger than the controls.
<img width="2400" height="1500" alt="ego_vs_random_controls" src="https://github.com/user-attachments/assets/5bba0f98-2c87-4347-a0a6-e7dc94a37fb2" />

### Incorrect sham control

An initial “sham ego” construction merely permuted the low-group examples.

This was mathematically invalid because:

[
\operatorname{mean}(L_\pi)=\operatorname{mean}(L)
]

so it produced the same ego vector.

That control was explicitly discarded.

### True randomized semantic sham

The activation examples were randomly partitioned into two groups, destroying the self/objective label.

This produced substantially weaker average behavior, although some individual sham directions still showed nonlinear trajectories.

-<img width="2400" height="1500" alt="real_ego_vs_true_sham" src="https://github.com/user-attachments/assets/2fc410ba-4cff-477b-9c36-191a0d5d95d9" />
--

# 13. Quantifying Inverted-U Shape

An inverted-U score was defined as the maximum elevation of the observed curve above the straight line connecting its endpoints.

For the original ego vector:

[
$\boxed{U_{\mathrm{ego}}\approx0.0082}$
]

while the initial random controls had:

[
$U_{\mathrm{random}}\approx0.000285$
]

The ego vector exceeded all initial controls.

---

# 14. Ego-Vector Robustness

Five independently bootstrapped ego vectors were constructed from the underlying self/objective activation examples.

Their inverted-U scores were:

[
[0.00815,;0.00547,;0.00700,;0.00758,;0.00762]
]

Mean:

[
$\boxed{0.007165}$
]

Standard deviation:

[
0.000922
]

Thus the nonlinear trajectory was not confined to a single exact estimate of the ego vector.
<img width="2400" height="1500" alt="ego_vector_robustness" src="https://github.com/user-attachments/assets/476b661e-6de6-4724-b3e3-12d6e14409d6" />

---

# 15. Random Null Robustness

Thirty random orthogonal control directions were then tested using the same experimental pipeline.

Their statistics were:

[
$\text{mean}=0.000168$
]

[
SD=0.000277
]

[
$\max=0.001093$
]

Every one of the five ego vectors exceeded every one of the 30 random-null U-scores.

The finite empirical comparison produced:

[
$\boxed{p\approx0.0323}$
]

The corresponding computed mean-ego/null z-score was approximately 25, although this is **not interpreted as a conventional Gaussian significance statistic**.
<img width="2400" height="1500" alt="ego_vs_random_controls" src="https://github.com/user-attachments/assets/af6e4096-b3f2-41aa-ac44-ae8d45208f72" />

---

# 16. Current Result

The current empirical picture is:

[$
\boxed{
\text{ego density}
\uparrow
\rightarrow
\Phi\uparrow
\rightarrow
\Phi_{\max}
\rightarrow
\Phi\downarrow
}$
]

and the inverted-U is substantially stronger for the tested ego vectors than for the random orthogonal null directions.

The current result should therefore be described as:

> **Within this GPT-2 Layer-8 / 3-node IIT-3 operationalization, self-referential activation steering produces a robust nonlinear relationship with IIT Φ, characterized by an initial increase followed by a decline at high steering strengths.**

---

# 17. Relation to the Awareness–Thought Space / DAM Framework

The theoretical manuscript describes a compositional dual-aspect-monist framework containing a psychophysically neutral Awareness Space, an awareness field, localized observers, thought vectors, and an Ego structure produced through thought stabilization/mental inertia.  

The manuscript currently defines Ego density as:

[
$\rho_E=\frac{I_E}{Kr^m}$
]

where $(I_E)$ represents accumulated thought intensity and (Kr^m) corresponds to the volume scale of the self-awareness region. 

The current empirical investigation does **not** attempt to validate the manuscript's singularity/collapse mechanism.

Instead, the working interpretation is narrower:

> A moderate degree of ego organization may support an observer's awareness/integration, while excessive ego organization eventually becomes counterproductive.

This motivates an empirical relationship of the form:

[
$\boxed{\Phi=f(\rho_E)}$
]

with (f) potentially exhibiting an inverted-U structure.

---

# 18. Important Limitations

This project does **not** demonstrate that GPT-2 is conscious.

It also does not establish that:

[
$\Phi=\text{phenomenal awareness}$
]

as a metaphysical fact.

The current result is dependent on several modeling choices:

* definition of the ego vector;
* choice of GPT-2 Layer 8;
* dimensionality and construction of the 3-node substrate;
* mapping from continuous activation space to a TPM;
* PyPhi's IIT-3 formalism;
* the selected prompt and test state.

The experiments therefore constitute a **computational test of a specific operationalization**, not yet a general law about human consciousness.

---

# 19. Planned Experiments

The next research phase will investigate:

1. **Different ego content** — whether Φ depends on the specific composition of the ego rather than only its magnitude.
2. **Positive vs defensive ego directions** — whether different forms of self-modeling affect Φ differently.
3. **Different ego vectors at different stages** — whether ego composition or trajectory through activation space matters.
4. **Multiple prompts and contexts**.
5. **Multiple GPT-2 layers**, while retaining Layer 8 as the primary theoretically motivated layer.
6. **Larger and more principled causal substrates**.
7. **Additional null/control directions**.
8. **Independent replication of the inverted-U trajectory**.
9. Investigation of the mechanistic reason Φ eventually declines at high ego density.

---

## Current Research Question

The central question has evolved from:

> **Does ego reduce consciousness?**

to:

> **Does ego-like self-organization initially increase integrated information, and does excessive ego density eventually become counterproductive?**

The current computational evidence is **consistent with an inverted-U relationship**.

The next stage is to determine whether that relationship reflects a genuine property of self-referential organization or an artifact of the GPT-2 → TPM → IIT operationalization.

---

## Citation / Conceptual Background

The theoretical framework motivating this work is described in:
[**Khurana, Sidharth, The Mechanics of Awareness-Thought Space.(2026)**](https://philpapers.org/rec/KHUTMO)

The manuscript develops the Awareness–Thought Space, compositional dual-aspect monism, thought-space geometry, awareness-field dynamics, observers, mental inertia, and Ego density. 

---

## Disclaimer

This repository contains **exploratory research code and preliminary computational results**. Results should not be interpreted as evidence that GPT-2 is conscious, that IIT has been empirically validated as a theory of phenomenal consciousness, or that the proposed DAM framework has been experimentally established.

The purpose of the repository is to make the experimental process **transparent, reproducible, falsifiable, and open to criticism**.
