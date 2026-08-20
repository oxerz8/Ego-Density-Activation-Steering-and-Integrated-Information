# ### 1. The Core Objective
# 
# * **Goal:** Test if self-referential ("I-position") thoughts form a distinct, dense geometric cluster inside a transformer's internal representations (the residual stream), validating the DAM prediction of an observer/ego concentration center.
# * **Model Used:** GPT-2 Small (12 layers, 768 dimensions per vector).
# 
# ---
# 
# ### 2. Experiment 1: Measuring Cluster Density (Cosine Similarity)
# 
# * **Setup:** Extracted the final-token residual stream vectors for **50 self-referential** sentences (e.g., *"I think therefore I am"*) and **50 objective** sentences (e.g., *"Water boils at 100 degrees"*).
# * **Findings:**
# * **Self-to-Self Similarity:** `0.9794` (Very dense cluster)
# * **Objective-to-Objective Similarity:** `0.9510` (More diffuse)
# * **Cross-Group Similarity:** `0.9475` (Separated in vector space)
# 
# 
# * **Takeaway:** Self-referential concepts form a tighter geometric pocket than general objective facts.
# 
# ---
# 
# ### 3. Experiment 2: 2D PCA Visualization
# 
# * **Setup:** Applied Principal Component Analysis (PCA) to project the 768-dimensional vectors into 2D.
# * **Findings:**
# * **PC1 (24.1% variance)** created a clean linear separation between the two groups.
# * Self-referential points clustered tightly on the right, while objective points scattered across the left.
# 
# 
# 
# ---
# 
# ### 4. Experiment 3: Layer-by-Layer Emergence
# 
# * **Setup:** Measured the similarity metrics across all 12 layers (Layer 0 to 11) to see *where* this separation happens.
# * **Findings:**
# * **Layers 0–2:** Vectors start undifferentiated (near 1.0 similarity).
# * **Layers 3–8:** The model actively pulls the concepts apart. Cross-group similarity drops to its lowest point (`~0.800`) at **Layer 8**, while self-referential vectors remain cohesive (`~0.905`).
# * **Layers 9–11:** Vectors compress back together for vocabulary projection.
# 
# 
# 
# ---
# 
# ### 5. Experiment 4: Layer 8 Circuit Analysis
# 
# * **Setup:** Inspected all 12 attention heads in Layer 8 to see which ones route attention from the final token back to the subject token (`"I"` vs. `"The"`).
# * **Findings:**
# * **Head 8.9 and Head 8.3** showed strong preference for self-referential tokens over objective subjects.
# * **Head 8.7** acted as a general attention sink (routing to the first token regardless of content).

# %%
import torch
import transformer_lens
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Load model
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

self_prompts = [
    "I think therefore I am.",
    "I feel incredibly happy today.",
    "My perspective on this is unique.",
    "I am aware of my own existence.",
    "My thoughts feel scattered today.",
    "I prefer coffee over tea.",
    "I remember my childhood clearly.",
    "I am trying to focus on my breath.",
    "My identity is constantly evolving.",
    "I feel a sense of calm.",
    "I doubt my own conclusions sometimes.",
    "I am observing my mind.",
    "My emotions fluctuate rapidly.",
    "I consider myself a curious person.",
    "I know what I want.",
    "I am experiencing this moment.",
    "My consciousness feels localized.",
    "I believe in my potential.",
    "I feel pain when I am injured.",
    "I trust my intuition.",
    "I am reflecting on my past.",
    "My sense of self is strong.",
    "I perceive the world through my senses.",
    "I am learning new things every day.",
    "I feel overwhelmed by the noise.",
    "My beliefs define my actions.",
    "I am searching for meaning.",
    "I experience time linearly.",
    "My body feels tired.",
    "I am capable of change.",
    "I recognize my own face in the mirror.",
    "I feel disconnected from reality.",
    "My desires drive my behavior.",
    "I am in control of my choices.",
    "I value my independence.",
    "I am conscious of my surroundings.",
    "My ego feels threatened.",
    "I am imagining a different future.",
    "I feel deeply connected to nature.",
    "My inner voice never stops.",
    "I am aware of my biases.",
    "I find joy in simple things.",
    "My existence feels profound.",
    "I am questioning my motives.",
    "I feel a profound sense of awe.",
    "My thoughts are my own.",
    "I am navigating my emotions.",
    "I sense a change in my mood.",
    "My perspective is limited.",
    "I feel at peace with myself."
]

obj_prompts = [
    "The sky appears blue today.",
    "Water boils at exactly 100 degrees.",
    "A standard triangle has three sides.",
    "Jupiter is the largest planet in our solar system.",
    "Oxygen has an atomic number of eight.",
    "The speed of light is constant in a vacuum.",
    "DNA carries genetic information.",
    "Gravity pulls objects toward the center of mass.",
    "The Pacific Ocean is the largest ocean.",
    "Photosynthesis requires carbon dioxide and water.",
    "Triangles have interior angles summing to 180 degrees.",
    "Iron oxidizes to form rust.",
    "The human skeleton has 206 bones.",
    "Sound travels faster in water than in air.",
    "Mount Everest is the highest peak on Earth.",
    "The moon orbits the Earth.",
    "Electrons have a negative charge.",
    "Rome is the capital of Italy.",
    "Water freezes at zero degrees Celsius.",
    "The Great Wall of China is a series of fortifications.",
    "Friction opposes motion.",
    "Saturn has extensive ring systems.",
    "The boiling point of water decreases at higher altitudes.",
    "Diamonds are made of carbon.",
    "The Sahara is the largest hot desert.",
    "Neurons transmit electrical signals.",
    "The Pythagorean theorem applies to right triangles.",
    "Helium is lighter than air.",
    "The Amazon River flows into the Atlantic Ocean.",
    "Metals conduct electricity.",
    "The Earth's core is primarily iron and nickel.",
    "Mars is known as the Red Planet.",
    "Antibiotics treat bacterial infections.",
    "The speed of sound is Mach 1.",
    "Pi is an irrational number.",
    "The human heart has four chambers.",
    "Plants produce oxygen as a byproduct.",
    "The Eiffel Tower is located in Paris.",
    "Volcanoes erupt magma.",
    "Kinetic energy is the energy of motion.",
    "The solar system belongs to the Milky Way galaxy.",
    "An octave consists of eight notes.",
    "Tides are caused by gravitational pull.",
    "Chlorophyll gives plants their green color.",
    "The square root of 64 is 8.",
    "Earthquakes are measured on the Richter scale.",
    "The human brain contains billions of neurons.",
    "Gold is a chemical element with the symbol Au.",
    "The atmosphere consists mostly of nitrogen.",
    "Light can act as a wave or a particle."
]

def get_residual_vector(prompt):
    # Run with cache to grab internal states
    logits, cache = model.run_with_cache(prompt)
    
    # Dynamically find the last layer index
    last_layer = model.cfg.n_layers - 1
    hook_name = f'blocks.{last_layer}.hook_resid_post'
    
    # Extract the residual stream after the final block, for the final token
    residual = cache[hook_name][0, -1, :].cpu().numpy()
    return residual

# 3. Extract Vectors
self_vectors = np.array([get_residual_vector(p) for p in self_prompts])
obj_vectors = np.array([get_residual_vector(p) for p in obj_prompts])

# 4. Measure Clustering (Cosine Similarity)
self_sim = cosine_similarity(self_vectors)
obj_sim = cosine_similarity(obj_vectors)
cross_sim = cosine_similarity(self_vectors, obj_vectors)

# Calculate means (excluding self-to-identical-self diagonals for within-group)
mean_self = self_sim[np.triu_indices_from(self_sim, k=1)].mean()
mean_obj = obj_sim[np.triu_indices_from(obj_sim, k=1)].mean()
mean_cross = cross_sim.mean()

print(f"Self-to-Self Mean Similarity: {mean_self:.4f}")
print(f"Objective-to-Objective Mean Similarity: {mean_obj:.4f}")
print(f"Cross-Group Mean Similarity: {mean_cross:.4f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Combine vectors and create labels
# vstack stacks the arrays vertically
all_vectors = np.vstack((self_vectors, obj_vectors))

# Create labels: 0 for Self-referential, 1 for Objective
labels = np.array([0] * len(self_vectors) + [1] * len(obj_vectors))

# 2. Perform PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(all_vectors)

# Separate the reduced vectors back into their groups for plotting
self_reduced = reduced_vectors[labels == 0]
obj_reduced = reduced_vectors[labels == 1]

# Get the variance explained by each component (good practice for papers)
explained_variance = pca.explained_variance_ratio_ * 100

# 3. Create the visualization
plt.figure(figsize=(8, 6), dpi=300) # dpi=300 ensures print-quality resolution

# Plot Self-Referential (I-Position) - using red circles
plt.scatter(self_reduced[:, 0], self_reduced[:, 1], 
            color='darkred', marker='o', alpha=0.7, s=50, 
            label='Self-Referential (I-Position)', edgecolors='black')

# Plot Objective - using blue triangles
plt.scatter(obj_reduced[:, 0], obj_reduced[:, 1], 
            color='steelblue', marker='^', alpha=0.7, s=50, 
            label='Objective (Factual)', edgecolors='black')

# Styling for academic presentation
plt.title('PCA of Residual Stream Activations:\nSelf-Referential vs. Objective Text (GPT-2)', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.1f}% Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.1f}% Variance)', fontsize=12)
plt.legend(loc='best', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.4)

# Adjust layout, save the figure, and display
plt.tight_layout()
plt.savefig('pca_thought_space_clustering.png')
print("Plot saved as 'pca_thought_space_clustering.png'")
plt.show()

# %% [markdown]
# So in the above example, there's a clean distinction between self referential thoughts and objectives ones. These are also visualizable in the above graph.

# %% [markdown]
# 

# %% [markdown]
# the complete script to extract and plot the layer-by-layer evolution of the residual stream activations. This will show exactly at which layer the Ego cluster separates from objective facts.

# %%
import numpy as np
import transformer_lens
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 1. Load Model
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

def get_layer_activations(prompts):
    """Extracts final-token residual vectors for every layer."""
    # Initialize dictionary to hold vectors for each layer
    layer_vectors = {layer: [] for layer in range(model.cfg.n_layers)}
    
    for prompt in prompts:
        # run_with_cache captures all internal activations
        _, cache = model.run_with_cache(prompt)
        
        # Extract the residual stream for the last token at every layer
        for layer in range(model.cfg.n_layers):
            hook_name = f'blocks.{layer}.hook_resid_post'
            residual = cache[hook_name][0, -1, :].cpu().numpy()
            layer_vectors[layer].append(residual)
            
    # Convert lists to numpy arrays
    for layer in range(model.cfg.n_layers):
        layer_vectors[layer] = np.array(layer_vectors[layer])
        
    return layer_vectors

# 2. Extract Activations (Assuming self_prompts and obj_prompts are loaded from earlier)
print("Extracting layer-by-layer activations... This might take a moment.")
self_layer_vecs = get_layer_activations(self_prompts)
obj_layer_vecs = get_layer_activations(obj_prompts)

# 3. Calculate Similarity Evolution
layers = list(range(model.cfg.n_layers))
self_means, obj_means, cross_means = [], [], []

for layer in layers:
    self_vecs = self_layer_vecs[layer]
    obj_vecs = obj_layer_vecs[layer]
    
    # Calculate similarity matrices
    self_sim = cosine_similarity(self_vecs)
    obj_sim = cosine_similarity(obj_vecs)
    cross_sim = cosine_similarity(self_vecs, obj_vecs)
    
    # Mean of upper triangle (ignoring self-identity diagonals)
    mean_self = self_sim[np.triu_indices_from(self_sim, k=1)].mean()
    mean_obj = obj_sim[np.triu_indices_from(obj_sim, k=1)].mean()
    mean_cross = cross_sim.mean()
    
    self_means.append(mean_self)
    obj_means.append(mean_obj)
    cross_means.append(mean_cross)

# 4. Visualize the Emergence
plt.figure(figsize=(10, 6), dpi=300)

plt.plot(layers, self_means, label='Self-to-Self (Ego Cluster)', color='darkred', marker='o', linewidth=2)
plt.plot(layers, obj_means, label='Objective-to-Objective', color='steelblue', marker='^', linewidth=2)
plt.plot(layers, cross_means, label='Cross-Group Similarity', color='gray', marker='s', linestyle='--', linewidth=2)

plt.title('Layer-by-Layer Emergence of the I-Position (GPT-2 Small)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Transformer Layer (0 to 11)', fontsize=12)
plt.ylabel('Mean Cosine Similarity', fontsize=12)
plt.xticks(layers)
plt.legend(loc='best', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('layer_evolution.png')
print("Plot saved as 'layer_evolution.png'")
plt.show()

# %% [markdown]
# Here is the code to inspect all 12 attention heads in Layer 8.
# 
# This script measures how strongly each head in Layer 8 routes attention from the final token back to the subject of the sentence (e.g., "I" / "My" in self-referential prompts vs. "The" / "Water" in objective prompts) and plots the comparison.

# %%
import numpy as np
import matplotlib.pyplot as plt
import transformer_lens

# 1. Load model (if not already loaded)
# model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

target_layer = 8
n_heads = model.cfg.n_heads  # 12 heads in GPT-2 small

def get_subject_attention_weights(prompts, layer=8):
    """
    Measures the attention weight from the last token back to the first token 
    (the subject: 'I', 'My', 'The', etc.) across all heads in a given layer.
    """
    head_attention = np.zeros((len(prompts), n_heads))
    
    for i, prompt in enumerate(prompts):
        _, cache = model.run_with_cache(prompt)
        
        # Attention pattern shape: [batch, n_heads, query_pos, key_pos]
        attn_pattern = cache[f'blocks.{layer}.attn.hook_pattern'][0]
        
        # Attention from the last token (query) to the first token (key: subject)
        # attn_pattern[:, -1, 0] gives weights for all 12 heads
        last_to_first = attn_pattern[:, -1, 0].cpu().numpy()
        head_attention[i] = last_to_first
        
    # Return the mean attention weight per head across all prompts
    return head_attention.mean(axis=0)

# 2. Extract attention profiles for Layer 8
print(f"Extracting attention patterns for Layer {target_layer}...")
self_head_attn = get_subject_attention_weights(self_prompts, layer=target_layer)
obj_head_attn = get_subject_attention_weights(obj_prompts, layer=target_layer)

# 3. Plot Comparison of Heads
head_indices = np.arange(n_heads)
width = 0.35

plt.figure(figsize=(10, 5), dpi=300)

plt.bar(head_indices - width/2, self_head_attn, width, 
        label='Self-Referential ("I" / "My")', color='darkred', alpha=0.85, edgecolor='black')
plt.bar(head_indices + width/2, obj_head_attn, width, 
        label='Objective ("The" / "Water" / etc.)', color='steelblue', alpha=0.85, edgecolor='black')

plt.title(f'Layer {target_layer} Attention Heads: Focus on Sentence Subject', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Attention Head Index (0 to 11)', fontsize=11)
plt.ylabel('Mean Attention Weight (Last Token -> Subject)', fontsize=11)
plt.xticks(head_indices, [f'H{i}' for i in head_indices])
plt.legend(loc='best', framealpha=0.9)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f'layer_{target_layer}_heads.png')
print(f"Plot saved as 'layer_{target_layer}_heads.png'")
plt.show()

