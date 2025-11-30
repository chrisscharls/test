# FeT (Federated Transformer) - Complete Workflow

## Table of Contents
1. [Model Initialization](#1-model-initialization)
2. [Data Preparation](#2-data-preparation)
3. [Forward Pass Workflow](#3-forward-pass-workflow)
4. [Training Loop](#4-training-loop)
5. [Privacy Mechanisms](#5-privacy-mechanisms)
6. [Evaluation & Inference](#6-evaluation--inference)

---

## 1. Model Initialization

### 1.1 Model Creation
```python
model = FeT(
    key_dims=[k1, k2, ..., kn],          # Key dimensions for each party
    data_dims=[d1, d2, ..., dn],          # Data dimensions for each party
    out_dim=n_classes,                    # Output dimension
    data_embed_dim=embed_dim,             # Embedding dimension
    num_heads=num_heads,                  # Multi-head attention heads
    dropout=0.1,                          # Dropout rate
    party_dropout=0.0,                    # Party dropout rate
    activation='gelu',                    # Activation function
    n_local_blocks=1,                     # Local self-attention blocks
    n_agg_blocks=1,                       # Aggregation attention blocks
    primary_party_id=0,                   # Primary party ID
    rep_noise=noise_scale,                # DP noise scale
    max_rep_norm=clip_norm,              # DP norm clipping
    enable_pe=True,                       # Enable positional encoding
    enable_dm=True                        # Enable dynamic masking
)
```

### 1.2 Component Initialization (Inside `__init__`)

**Step 1.1: Activation Function Setup**
- Converts string to PyTorch activation (ReLU, GELU, LeakyReLU)

**Step 1.2: Positional Encodings** (Lines 254-258)
- Creates one `LearnableFourierPositionalEncoding` per party
- Each encodes spatial/temporal key information
- Parameters: `G=1, M=key_dim, F_dim=data_embed_dim, H_dim=2*data_embed_dim, D=data_embed_dim, gamma=0.05`

**Step 1.3: Dynamic Mask Layers** (Lines 267-281)
- **Primary party**: Identity layer (no masking)
- **Secondary parties**: MLP `key_dim*k → 2*key_dim*k → k`
- Generates mask values to weight key-value pairs

**Step 1.4: Data Embeddings** (Lines 284-287)
- Linear layers: `(key_dim + data_dim) → data_embed_dim`
- One per party
- Concatenates key and data before embedding

**Step 1.5: Self-Attention Chains** (Lines 291-299)
- One `SelfAttnChain` per party
- Each has `n_local_blocks` transformer blocks
- Structure: Self-Attn → FFN with residuals

**Step 1.6: Aggregation Attention** (Lines 303-304)
- Single `AggAttnChain` on primary party
- Has `n_agg_blocks` blocks
- Structure: Self-Attn → Cross-Attn → FFN

**Step 1.7: Output Layer** (Line 307)
- Linear: `data_embed_dim → out_dim`

---

## 2. Data Preparation

### 2.1 Dataset Structure
Each sample consists of:
- **Keys**: `[(k1, k2, ..., kn)]` - Fuzzy identifiers for each party
- **Data**: `[(X1, X2, ..., Xn)]` - Features for each party
- **Labels**: `y` - Target values

### 2.2 Data Loading
```python
# From DataLoader
Xs, y = next(iter(train_loader))
# Xs format: [(key_0, data_0), (key_1, data_1), ...]
# Each key_X[i] = (tensor(batch, key_dim), tensor(batch, data_dim))
```

### 2.3 Preprocessing (Fit.py:33-49)
```python
def preprocess_Xs_y(Xs, y, device, task, has_key=True):
    # Move to device
    Xs = [(Xi[0].float().to(device), Xi[1].float().to(device)) for Xi in Xs]
    y = y.to(device).flatten()
    
    # Convert labels based on task
    y = y.long() if task == 'multi-cls' else y.float()
    return Xs, y
```

---

## 3. Forward Pass Workflow

### 3.1 Input Validation (Line 341)
```python
self.check_args(key_Xs)
```
- Validates tensor types and shapes
- Ensures batch sizes match across parties
- Checks dimensions match expected `key_dims` and `data_dims`

### 3.2 Input Shape Normalization (Lines 343-351)
```python
# Ensure sequence dimension exists: (batch, seq_len, features)
if len(key_X[0].shape) == 2:
    keys.append(key_X[0].unsqueeze(1))  # Add seq_len=1
    Xs.append(key_X[1].unsqueeze(1))
else:
    keys.append(key_X[0])
    Xs.append(key_X[1])
```
**Result**: All tensors have shape `(batch_size, seq_len, features)`

### 3.3 Dynamic Masking (Lines 354-362)
```python
for i in range(n_parties):
    if not enable_dm or i == primary_party_id:
        mask = None
    else:
        # Generate mask from keys
        mask = dynamic_mask_layers[i](keys[i].reshape(batch, -1))
    masks.append(mask)
```
**Purpose**: Secondary parties learn which key-value pairs to attend to
**Output**: `masks[i]` = `None` (primary) or `tensor(batch, k)` (secondary)

### 3.4 Embedding & Positional Encoding (Lines 365-378)

**Step 3.4.1: Concatenate Key and Data**
```python
key_Xs = [torch.cat([keys[i], Xs[i]], dim=-1) for i in range(n_parties)]
# Shape: (batch, seq_len, key_dim + data_dim)
```

**Step 3.4.2: Embed to data_embed_dim**
```python
key_X_embed = data_embeddings[i](key_Xs[i])
# Shape: (batch, seq_len, data_embed_dim)
```

**Step 3.4.3: Add Positional Encoding** (if `enable_pe=True`)
```python
if enable_pe:
    pe = positional_encodings[i](keys[i].view(-1, 1, key_dim))
    # pe shape: (batch*seq_len, data_embed_dim)
    key_X_embed += pe.view(key_X_embed.shape)
```
**Purpose**: Encodes spatial/temporal relationships in keys

**Result**: `key_X_embeds[i]` shape `(batch, seq_len, data_embed_dim)`

### 3.5 Local Self-Attention (Lines 381-383)
```python
for i in range(n_parties):
    if i != primary_party_id:
        # Secondary parties: process through self-attention
        key_X_embeds[i] = self_attns[i](
            key_X_embeds[i], 
            need_weights=visualize,
            key_padding_mask=masks[i]
        )
    # Primary party: pass through unchanged (for now)
```
**Process** (for each secondary party):
1. **Self-Attention Block** (n_local_blocks times):
   - `LayerNorm(x) → SelfAttention → Add(x)`
   - `LayerNorm(x) → FFN → Add(x)`
2. **Random Masking** (if training):
   - Randomly masks some positions with `-inf`
   - Forces model to handle missing data

**Result**: Secondary parties have processed embeddings

### 3.6 Party Dropout (Lines 396-410)
```python
if training and party_dropout > 0:
    # Randomly drop some secondary parties
    n_drop = int((n_parties - 1) * party_dropout)
    drop_indices = torch.randperm(n_parties - 1)[:n_drop]
    
    # Zero out dropped parties
    drop_mask = torch.ones(n_parties - 1)
    drop_mask[drop_indices] = 0.0
    secondary_key_X_embeds = [drop_mask[i] * secondary_key_X_embeds[i]
                              for i in range(n_parties - 1)]
```
**Purpose**: Improves robustness to missing parties during inference

### 3.7 Cut Layer & Privacy Mechanisms (Lines 412-433)

**Step 3.7.1: Extract Representations**
```python
primary_key_X_embed = key_X_embeds[primary_party_id]
secondary_key_X_embeds = [key_X_embeds[i] 
                          for i in range(n_parties) 
                          if i != primary_party_id]
```

**Step 3.7.2: Apply Differential Privacy** (if `rep_noise` and `max_rep_norm` set)

**A. Norm Clipping per Party:**
```python
max_rep_norm_per_party = max_rep_norm / (n_parties - 1)
for secondary_key_X_embed in secondary_key_X_embeds:
    # Reduce norm with tanh
    cut_layer_i = torch.tanh(secondary_key_X_embed)
    cut_layer_i_flat = cut_layer_i.reshape(batch, -1)
    
    # Compute per-sample L2 norm
    per_sample_norm = torch.norm(cut_layer_i_flat, dim=1, p=2)
    
    # Clip to max norm
    clip_coef = max_rep_norm_per_party / (per_sample_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1)
    rep = cut_layer_i_flat * clip_coef.unsqueeze(-1)
    secondary_reps.append(rep)
```

**B. Aggregate with Sum:**
```python
cut_layer_key_X_flat = torch.sum(torch.stack(secondary_reps), dim=0)
```

**C. Add Gaussian Noise:**
```python
noise = torch.normal(0, rep_noise * max_rep_norm, cut_layer_key_X_flat.shape)
cut_layer_key_X_flat += noise
cut_layer_key_X = cut_layer_key_X_flat.reshape(batch, -1, data_embed_dim)
```

**Step 3.7.3: Simple Aggregation** (if no DP)
```python
cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)
cut_layer_key_X /= (n_parties - n_drop_parties - 1)  # Average
```

**Note**: This is where MPC should be implemented (currently TODO)

### 3.8 Aggregation Attention (Lines 436-438)
```python
agg_key_X_embed = agg_attn(
    primary_key_X_embed,           # Query: primary party
    cut_layer_key_X,                # Key/Value: aggregated secondary parties
    need_weights=visualize,
    key_padding_mask=masks[primary_party_id]
)
```

**Process** (n_agg_blocks times):
1. **Self-Attention on Primary:**
   - `LayerNorm(x) → SelfAttention(primary, primary) → Add(x)`
2. **Cross-Attention:**
   - `LayerNorm(x) → CrossAttention(primary, aggregated_secondary) → Add(x)`
3. **Feed-Forward:**
   - `LayerNorm(x) → FFN → Add(x)`

**Purpose**: Primary party attends to aggregated secondary representations

### 3.9 Output Generation (Lines 444-447)
```python
# Flatten sequence dimension
output = output_layer(agg_key_X_embed.reshape(batch, -1))
# Shape: (batch, out_dim)

# Apply output activation if specified
if out_activation is not None:
    output = out_activation(output)
```

**Output Shapes**:
- Regression: `(batch, 1)`
- Binary Classification: `(batch, 1)` with sigmoid
- Multi-class: `(batch, n_classes)` with logits

---

## 4. Training Loop

### 4.1 Epoch Loop (Fit.py:98-281)

**For each epoch:**

#### 4.1.1 Training Phase (Lines 99-124)
```python
model.train()
for Xs, y in train_loader:
    # Preprocess
    Xs, y = preprocess_Xs_y(Xs, y, device, task, has_key=True)
    
    # Forward pass
    optimizer.zero_grad()
    y_pred = model(Xs)
    
    # Compute loss
    y_pred = y_pred.flatten() if task in ['reg', 'bin-cls'] else y_pred
    loss = loss_fn(y_pred, y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Collect predictions for metrics
    train_pred_y = torch.cat([train_pred_y, y_pred], dim=0)
    train_y = torch.cat([train_y, y], dim=0)
```

**Loss Functions**:
- Regression: `nn.MSELoss()`
- Binary Classification: `nn.BCELoss()`
- Multi-class: `nn.CrossEntropyLoss()`

#### 4.1.2 Validation Phase (Lines 157-205)
```python
model.eval()
with torch.no_grad():
    for Xs, y in val_loader:
        Xs, y = preprocess_Xs_y(Xs, y, device, task, has_key=True)
        y_pred = model(Xs)
        # ... compute metrics
```

#### 4.1.3 Test Phase (Lines 211-262)
```python
model.eval()
with torch.no_grad():
    for Xs, y in test_loader:
        # Similar to validation
```

#### 4.1.4 Positional Encoding Averaging (Lines 278-281)
```python
if epoch % average_pe_freq == 0 and epoch != 0:
    model.average_pe_()
```
**Purpose**: Synchronizes positional encodings across parties

---

## 5. Privacy Mechanisms

### 5.1 Differential Privacy Components

**A. Norm Clipping** (Lines 412-423)
- Limits L2 norm of each party's representation
- Per-party limit: `max_rep_norm / (n_parties - 1)`
- Prevents any single party from dominating

**B. Gaussian Noise** (Lines 428-430)
- Adds noise: `N(0, rep_noise * max_rep_norm)`
- Provides (ε, δ)-differential privacy
- Noise scale can be calibrated using `GaussianMechanism.calibrateAnalyticGaussianMechanism()`

**C. Dynamic Masking** (Lines 354-362)
- Secondary parties learn which keys to mask
- Reduces information leakage
- Mask values control attention weights

### 5.2 Privacy Guarantees

**Current Implementation**:
- ✅ Differential Privacy (noise + clipping)
- ❌ Secure Multi-Party Computation (TODO - line 390)

**Privacy Budget**:
- `rep_noise`: Controls ε (epsilon)
- `max_rep_norm`: Controls sensitivity (affects ε, δ)
- Can be calibrated using analytic Gaussian mechanism

---

## 6. Evaluation & Inference

### 6.1 Inference Mode
```python
model.eval()
with torch.no_grad():
    y_pred = model(Xs)
```

**Differences from Training**:
- No random masking in self-attention
- No party dropout
- No gradient computation
- Batch normalization in eval mode

### 6.2 Visualization Methods

**A. Positional Encoding Visualization** (Lines 465-500)
```python
model.visualize_positional_encoding(
    pivot=(0, 0),
    sample_size=50000,
    scale=1.0,
    save_path='pe_vis.png'
)
```
- Shows how positional encoding varies with key values
- Visualizes distance from pivot point

**B. Dynamic Mask Visualization** (Lines 545-649)
```python
model.visualize_dynamic_mask(
    key_Xs,
    key_scalers=scalers,
    title='Dynamic Mask',
    save_path='mask_vis.png'
)
```
- Visualizes mask values for secondary party keys
- Uses PCA for high-dimensional keys

**C. Attention Weight Visualization**
```python
y_pred = model(Xs, visualize=True)
model.self_attns[i].visualize_attention(head=None)
```
- Shows attention patterns in self-attention blocks

---

## 7. Complete Data Flow Diagram

```
INPUT: [(key_0, data_0), (key_1, data_1), ..., (key_n, data_n)]
  │
  ├─► Input Validation
  │
  ├─► Shape Normalization → (batch, seq_len, features)
  │
  ├─► Dynamic Masking (Secondary Parties Only)
  │   └─► mask = MLP(keys) → attention weights
  │
  ├─► Embedding Layer
  │   ├─► Concatenate: [key, data] → (key_dim + data_dim)
  │   └─► Linear: (key_dim + data_dim) → data_embed_dim
  │
  ├─► Positional Encoding (if enabled)
  │   └─► LearnableFourierPE(keys) → add to embeddings
  │
  ├─► Local Self-Attention (Secondary Parties)
  │   ├─► SelfAttnChain: n_local_blocks
  │   │   ├─► Self-Attention
  │   │   └─► FFN
  │   └─► Primary Party: Pass through unchanged
  │
  ├─► Party Dropout (Training Only)
  │   └─► Randomly zero out some secondary parties
  │
  ├─► Cut Layer & Privacy
  │   ├─► Extract secondary representations
  │   ├─► Norm Clipping (if DP enabled)
  │   ├─► Aggregate: sum(secondary_reps)
  │   └─► Add Gaussian Noise (if DP enabled)
  │
  ├─► Aggregation Attention (Primary Party)
  │   ├─► Self-Attention on primary
  │   ├─► Cross-Attention: primary → aggregated_secondary
  │   └─► FFN
  │
  ├─► Output Layer
  │   └─► Linear: data_embed_dim → out_dim
  │
  └─► OUTPUT: (batch, out_dim)
```

---

## 8. Key Design Decisions

### 8.1 Why Transformer Architecture?
- **Attention Mechanism**: Captures complex relationships between keys and features
- **Scalability**: Handles variable numbers of parties
- **Flexibility**: Works with fuzzy identifiers (imperfect key matching)

### 8.2 Why Positional Encoding?
- **Spatial Awareness**: Encodes relationships in key space
- **Fuzzy Matching**: Helps match similar but not identical keys
- **Learnable**: Adapts to data distribution

### 8.3 Why Dynamic Masking?
- **Selective Attention**: Focuses on relevant key-value pairs
- **Privacy**: Reduces information leakage
- **Efficiency**: Reduces computational cost

### 8.4 Why Cut Layer?
- **Privacy**: Limits information shared between parties
- **Differential Privacy**: Enables formal privacy guarantees
- **Aggregation**: Combines multi-party information securely

---

## 9. Training Configuration Example

```python
# Model
model = FeT(
    key_dims=[5, 5],              # 2 parties, 5-dim keys each
    data_dims=[100, 150],          # Different feature dimensions
    out_dim=2,                     # Binary classification
    data_embed_dim=128,
    num_heads=4,
    dropout=0.1,
    party_dropout=0.1,             # 10% party dropout
    n_local_blocks=2,
    n_agg_blocks=2,
    primary_party_id=0,
    rep_noise=0.01,                # DP noise
    max_rep_norm=1.0,              # DP clipping
    enable_pe=True,
    enable_dm=True
)

# Training
optimizer = optim.Lamb(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# Train
fit(
    model, optimizer, loss_fn, metric_fn,
    train_loader, val_loader, test_loader,
    epochs=100,
    task='bin-cls',
    average_pe_freq=10  # Average PE every 10 epochs
)
```

---

## 10. Summary

**FeT Workflow Summary**:
1. **Initialize** model with party-specific components
2. **Prepare** data with keys and features per party
3. **Embed** and encode with positional information
4. **Process** locally with self-attention (secondary parties)
5. **Aggregate** with privacy mechanisms (noise + clipping)
6. **Attend** to aggregated info (primary party)
7. **Predict** with output layer
8. **Train** with standard backpropagation
9. **Evaluate** with privacy-preserving inference

**Privacy Features**:
- ✅ Differential Privacy (noise + clipping)
- ✅ Dynamic Masking
- ✅ Party Dropout
- ❌ MPC (planned, not implemented)

**Key Innovation**: Transformer architecture for fuzzy vertical federated learning with strong privacy guarantees.

