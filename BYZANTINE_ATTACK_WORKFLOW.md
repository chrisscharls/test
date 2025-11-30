# Byzantine Attacks on FeT - Complete Workflow

## Table of Contents
1. [Overview](#1-overview)
2. [Attack Strategies](#2-attack-strategies)
3. [Attack Implementation](#3-attack-implementation)
4. [Attack Workflow](#4-attack-workflow)
5. [Integration with FeT](#5-integration-with-fet)
6. [Defense Mechanisms](#6-defense-mechanisms)
7. [Usage Examples](#7-usage-examples)

---

## 1. Overview

### 1.1 What are Byzantine Attacks?

**Byzantine attacks** in federated learning refer to malicious parties that send corrupted or adversarial updates to disrupt the training process. In FeT, these attacks occur at the **aggregation phase** where secondary parties send their representations to be aggregated.

### 1.2 Attack Point in FeT

The attack occurs at the **cut layer aggregation** step (lines 412-433 in FeT.py), where:
- Secondary parties send their processed representations
- Representations are aggregated (summed)
- Primary party uses aggregated representation for final prediction

```
Normal Flow:
Secondary Party 1 → Representation 1 ┐
Secondary Party 2 → Representation 2 ├─► Aggregate ─► Primary Party
Secondary Party 3 → Representation 3 ┘

Byzantine Attack:
Secondary Party 1 → Representation 1 ┐
Secondary Party 2 → CORRUPTED Rep 2 ├─► Aggregate ─► Primary Party (POISONED)
Secondary Party 3 → Representation 3 ┘
```

### 1.3 Attack Goals

1. **Model Poisoning**: Degrade model performance
2. **Training Disruption**: Prevent convergence
3. **Backdoor Injection**: Create hidden vulnerabilities
4. **Privacy Violation**: Extract information from other parties

---

## 2. Attack Strategies

### 2.1 Random Noise Attack
**Strategy**: Add random Gaussian noise to representation

**Formula**: `rep' = rep + N(0, attack_strength * std(rep))`

**Effect**: 
- Introduces random errors
- Reduces signal-to-noise ratio
- Degrades model accuracy

**Code**:
```python
def _random_noise(self, rep: torch.Tensor) -> torch.Tensor:
    noise_scale = self.attack_strength * rep.std().item()
    noise = torch.randn_like(rep) * noise_scale
    return rep + noise
```

### 2.2 Sign Flip Attack
**Strategy**: Flip the sign of the representation

**Formula**: `rep' = -attack_strength * rep`

**Effect**:
- Completely reverses the gradient direction
- Causes model to learn opposite patterns
- Very disruptive to training

**Code**:
```python
def _sign_flip(self, rep: torch.Tensor) -> torch.Tensor:
    return -self.attack_strength * rep
```

### 2.3 Scale Up Attack
**Strategy**: Scale up representation to dominate aggregation

**Formula**: `rep' = attack_strength * rep`

**Effect**:
- Malicious party dominates the aggregated representation
- Can overwhelm honest parties' contributions
- Creates bias toward malicious party's data distribution

**Code**:
```python
def _scale_up(self, rep: torch.Tensor) -> torch.Tensor:
    return self.attack_strength * rep
```

### 2.4 Zero Out Attack
**Strategy**: Send zero representation (dropout attack)

**Formula**: `rep' = 0`

**Effect**:
- Removes party's contribution entirely
- Simulates party dropout/failure
- Can be used for denial-of-service

**Code**:
```python
def _zero_out(self, rep: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(rep)
```

### 2.5 Adversarial Attack
**Strategy**: Create adversarial representation that maximizes disruption

**Formula**: `rep' = rep + attack_strength * sign(rep) * ||rep||`

**Effect**:
- Targets maximum gradient disruption
- Can cause training instability
- Harder to detect than random noise

**Code**:
```python
def _adversarial(self, rep: torch.Tensor) -> torch.Tensor:
    adversarial_direction = torch.sign(rep) * self.attack_strength
    return rep + adversarial_direction * rep.norm()
```

### 2.6 Gradient Scaling Attack
**Strategy**: Scale representation to create large gradient updates

**Formula**: `rep' = attack_strength * rep`

**Effect**:
- Similar to scale-up but focused on gradient magnitude
- Can cause exploding gradients
- Disrupts optimization process

---

## 3. Attack Implementation

### 3.1 ByzantineAttacker Class

```python
class ByzantineAttacker:
    def __init__(
        self,
        strategy: AttackStrategy,
        attack_strength: float = 1.0,
        malicious_parties: Optional[List[int]] = None,
        random_seed: Optional[int] = None
    ):
        self.strategy = strategy
        self.attack_strength = attack_strength
        self.malicious_parties = malicious_parties
        self.random_seed = random_seed
```

**Parameters**:
- `strategy`: Attack type (RANDOM_NOISE, SIGN_FLIP, etc.)
- `attack_strength`: Magnitude of attack (0.0 to 1.0+)
- `malicious_parties`: Which parties are malicious (None = all secondary)
- `random_seed`: For reproducibility

### 3.2 Attack Application

```python
def attack(
    self,
    representations: List[torch.Tensor],
    malicious_party_indices: List[int],
    primary_party_id: int = 0
) -> List[torch.Tensor]:
    """Apply attack to specified party representations"""
    corrupted_reps = [rep.clone() for rep in representations]
    
    for idx in malicious_party_indices:
        if idx < len(corrupted_reps):
            corrupted_reps[idx] = self._apply_strategy(corrupted_reps[idx])
    
    return corrupted_reps
```

---

## 4. Attack Workflow

### 4.1 Complete Attack Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FeT Forward Pass                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  1. Input Validation & Prep       │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  2. Embedding & Positional Enc.   │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  3. Local Self-Attention          │
        │     (Secondary Parties)           │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  4. Extract Representations      │
        │     Primary: rep_primary         │
        │     Secondary: [rep_1, rep_2, ...]│
        └───────────────────────────────────┘
                            │
                            ▼
        ╔═══════════════════════════════════╗
        ║  5. BYZANTINE ATTACK INJECTION   ║
        ║                                   ║
        ║  for malicious_party in           ║
        ║    malicious_parties:             ║
        ║      rep[malicious_party] =       ║
        ║        attack(rep[malicious_party])║
        ╚═══════════════════════════════════╝
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  6. Privacy Mechanisms            │
        │     (DP Noise + Clipping)         │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  7. Aggregation                   │
        │     aggregated = sum(secondary_reps)│
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  8. Aggregation Attention         │
        │     (Primary attends to aggregated)│
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  9. Output Layer                  │
        │     output = Linear(agg_rep)      │
        └───────────────────────────────────┘
                            │
                            ▼
                    CORRUPTED OUTPUT
```

### 4.2 Step-by-Step Attack Process

#### Step 1: Normal Processing (Lines 1-4)
- All parties process their data normally
- Secondary parties compute representations through self-attention
- No attack yet

#### Step 2: Representation Extraction
```python
# Extract secondary party representations
secondary_key_X_embeds = [
    key_X_embeds[i] 
    for i in range(n_parties) 
    if i != primary_party_id
]
```

#### Step 3: Attack Injection Point
```python
# ========== BYZANTINE ATTACK INJECTION POINT ==========
if attacker.strategy != AttackStrategy.NONE:
    # Determine malicious parties
    malicious_indices = [0, 1]  # Example: parties 1 and 2 are malicious
    
    # Apply attack
    secondary_key_X_embeds = attacker.attack(
        secondary_key_X_embeds,
        malicious_indices,
        primary_party_id
    )
# =====================================================
```

**What happens**:
1. Identify which secondary parties are malicious
2. For each malicious party, apply attack strategy
3. Corrupt their representation
4. Continue with corrupted representations

#### Step 4: Aggregation with Corrupted Data
```python
# Aggregate corrupted representations
cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)
```

**Impact**:
- Corrupted representations are included in sum
- Primary party receives poisoned aggregated representation
- Model learns from corrupted data

#### Step 5: Model Prediction
- Primary party attends to corrupted aggregated representation
- Output is computed from poisoned data
- Loss is computed with corrupted predictions
- Gradients propagate back through corrupted data

### 4.3 Attack Impact Propagation

```
Corrupted Representation
    │
    ├─► Aggregation (sum includes corrupted rep)
    │
    ├─► Primary Party Attention (attends to corrupted aggregate)
    │
    ├─► Output Layer (predicts from corrupted features)
    │
    ├─► Loss Computation (high loss due to corruption)
    │
    └─► Backward Pass (gradients propagate through corruption)
        │
        └─► Model Parameters (updated with corrupted gradients)
            │
            └─► Next Iteration (model is now poisoned)
```

---

## 5. Integration with FeT

### 5.1 ByzantineFeT Wrapper

The `ByzantineFeT` class wraps the original FeT model to inject attacks:

```python
class ByzantineFeT(nn.Module):
    def __init__(
        self,
        fet_model: nn.Module,
        attacker: Optional[ByzantineAttacker] = None,
        attack_probability: float = 0.0
    ):
        self.fet_model = fet_model
        self.attacker = attacker
        self.attack_probability = attack_probability
```

### 5.2 Attack Injection Method

The wrapper intercepts the forward pass at the aggregation step:

```python
def _inject_attack_in_forward(self, key_Xs, visualize=False):
    # ... normal FeT processing ...
    
    # Extract secondary representations
    secondary_key_X_embeds = [...]
    
    # ========== ATTACK INJECTION ==========
    if self.attacker.strategy != AttackStrategy.NONE:
        malicious_indices = [...]  # Determine malicious parties
        secondary_key_X_embeds = self.attacker.attack(
            secondary_key_X_embeds,
            malicious_indices,
            primary_party_id
        )
    # =====================================
    
    # Continue with aggregation
    cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)
    # ... rest of forward pass ...
```

---

## 6. Defense Mechanisms

### 6.1 Existing FeT Defenses

**A. Differential Privacy** (Lines 412-423)
- Norm clipping limits representation magnitude
- Gaussian noise adds randomness
- Reduces impact of single malicious party

**B. Party Dropout** (Lines 396-410)
- Randomly drops parties during training
- Can drop malicious parties by chance
- But not a reliable defense

### 6.2 Additional Defense Strategies

**A. Robust Aggregation**
```python
# Instead of simple sum, use median or trimmed mean
def robust_aggregate(representations):
    # Median aggregation (more robust to outliers)
    return torch.median(torch.stack(representations), dim=0)[0]
    
    # Or trimmed mean (remove top/bottom k%)
    sorted_reps = torch.sort(torch.stack(representations), dim=0)[0]
    trim_k = len(representations) // 10
    return sorted_reps[trim_k:-trim_k].mean(dim=0)
```

**B. Outlier Detection**
```python
def detect_outliers(representations, threshold=3.0):
    # Compute mean and std
    mean_rep = torch.stack(representations).mean(dim=0)
    std_rep = torch.stack(representations).std(dim=0)
    
    # Flag outliers
    outliers = []
    for i, rep in enumerate(representations):
        z_score = (rep - mean_rep) / (std_rep + 1e-8)
        if z_score.abs().max() > threshold:
            outliers.append(i)
    
    return outliers
```

**C. Gradient Clipping**
```python
# Clip gradients to prevent large updates from corrupted data
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**D. Reputation System**
```python
# Track party contributions over time
party_reputation = {i: 1.0 for i in range(n_parties)}

# Update reputation based on contribution quality
def update_reputation(party_id, contribution_quality):
    party_reputation[party_id] = (
        0.9 * party_reputation[party_id] + 
        0.1 * contribution_quality
    )
    
# Weight aggregation by reputation
weighted_sum = sum(
    rep * party_reputation[i] 
    for i, rep in enumerate(representations)
)
```

---

## 7. Usage Examples

### 7.1 Basic Attack Setup

```python
from src.model.FeT import FeT
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy

# Create base FeT model
model = FeT(
    key_dims=[5, 5, 5],
    data_dims=[100, 150, 200],
    out_dim=2,
    data_embed_dim=128,
    # ... other parameters ...
)

# Create attacker
attacker = ByzantineAttacker(
    strategy=AttackStrategy.RANDOM_NOISE,
    attack_strength=2.0,
    malicious_parties=[1, 2],  # Parties 1 and 2 are malicious
    random_seed=42
)

# Wrap model with attacker
byzantine_model = ByzantineFeT(
    fet_model=model,
    attacker=attacker,
    attack_probability=1.0  # Attack every batch
)

# Train with attacks
for epoch in range(epochs):
    for Xs, y in train_loader:
        y_pred = byzantine_model(Xs)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
```

### 7.2 Different Attack Strategies

```python
# Random noise attack
attacker_noise = ByzantineAttacker(
    strategy=AttackStrategy.RANDOM_NOISE,
    attack_strength=1.5
)

# Sign flip attack
attacker_flip = ByzantineAttacker(
    strategy=AttackStrategy.SIGN_FLIP,
    attack_strength=1.0
)

# Scale up attack
attacker_scale = ByzantineAttacker(
    strategy=AttackStrategy.SCALE_UP,
    attack_strength=10.0  # 10x scaling
)

# Zero out attack
attacker_zero = ByzantineAttacker(
    strategy=AttackStrategy.ZERO_OUT,
    attack_strength=1.0  # Not used for zero out
)
```

### 7.3 Partial Attack (Some Parties Malicious)

```python
# Only party 1 is malicious
attacker = ByzantineAttacker(
    strategy=AttackStrategy.SIGN_FLIP,
    attack_strength=1.0,
    malicious_parties=[1]  # Only secondary party 1
)

# Parties 1 and 2 are malicious
attacker = ByzantineAttacker(
    strategy=AttackStrategy.SCALE_UP,
    attack_strength=5.0,
    malicious_parties=[1, 2]
)
```

### 7.4 Probabilistic Attacks

```python
# Attack with 50% probability per batch
byzantine_model = ByzantineFeT(
    fet_model=model,
    attacker=attacker,
    attack_probability=0.5  # 50% chance of attack
)
```

### 7.5 Attack Statistics

```python
# After training, check attack statistics
stats = byzantine_model.get_attack_stats()
print(f"Attacks performed: {stats['attack_count']}")
print(f"Total batches: {stats['total_batches']}")
print(f"Attack rate: {stats['attack_rate']:.2%}")
print(f"Strategy: {stats['strategy']}")
```

### 7.6 Comparison: Normal vs. Attacked Training

```python
# Normal training
normal_model = FeT(...)
normal_losses = train(normal_model, train_loader)

# Attacked training
attacker = ByzantineAttacker(
    strategy=AttackStrategy.SIGN_FLIP,
    attack_strength=1.0
)
byzantine_model = ByzantineFeT(normal_model, attacker)
attacked_losses = train(byzantine_model, train_loader)

# Compare
import matplotlib.pyplot as plt
plt.plot(normal_losses, label='Normal')
plt.plot(attacked_losses, label='Byzantine Attack')
plt.legend()
plt.show()
```

### 7.7 Testing Defense Mechanisms

```python
# Test with different attack strengths
attack_strengths = [0.5, 1.0, 2.0, 5.0, 10.0]
results = []

for strength in attack_strengths:
    attacker = ByzantineAttacker(
        strategy=AttackStrategy.RANDOM_NOISE,
        attack_strength=strength
    )
    model = ByzantineFeT(base_model, attacker)
    final_acc = train_and_evaluate(model, train_loader, test_loader)
    results.append((strength, final_acc))

# Plot attack strength vs. accuracy
strengths, accs = zip(*results)
plt.plot(strengths, accs, 'o-')
plt.xlabel('Attack Strength')
plt.ylabel('Test Accuracy')
plt.title('Model Robustness to Byzantine Attacks')
plt.show()
```

---

## 8. Attack Workflow Summary

### 8.1 Attack Timeline

```
Epoch 1, Batch 1:
  └─► Normal processing
  └─► Attack injected at aggregation
  └─► Corrupted output
  └─► Model parameters updated with corrupted gradients

Epoch 1, Batch 2:
  └─► Model already slightly poisoned
  └─► Another attack
  └─► Further corruption

...

Epoch N:
  └─► Model significantly degraded
  └─► Poor performance on test set
```

### 8.2 Attack Impact Metrics

**Metrics to track**:
1. **Training Loss**: Should increase with attacks
2. **Test Accuracy**: Should decrease with attacks
3. **Gradient Norm**: May increase (exploding gradients)
4. **Representation Distance**: Distance between normal and attacked representations
5. **Convergence Rate**: Slower or no convergence

### 8.3 Key Points

1. **Attack Location**: Cut layer aggregation (line ~425 in FeT)
2. **Attack Timing**: Before aggregation, after local processing
3. **Attack Scope**: Can target specific parties or all secondary parties
4. **Attack Persistence**: Effects accumulate over training
5. **Attack Detection**: Monitor training metrics for anomalies

---

## 9. Code Structure

```
src/
├── attack/
│   ├── __init__.py
│   └── ByzantineAttack.py
│       ├── AttackStrategy (Enum)
│       ├── ByzantineAttacker (Class)
│       └── ByzantineFeT (Wrapper Class)
│
└── model/
    └── FeT.py (Original model)
```

---

## 10. Conclusion

Byzantine attacks on FeT occur at the **aggregation phase** where secondary parties send representations. The attack workflow:

1. **Normal Processing**: All parties process data normally
2. **Attack Injection**: Malicious parties corrupt their representations
3. **Aggregation**: Corrupted representations are aggregated
4. **Propagation**: Corruption affects primary party and model output
5. **Accumulation**: Effects accumulate over training epochs

**Defense strategies** include:
- Robust aggregation (median, trimmed mean)
- Outlier detection
- Gradient clipping
- Reputation systems
- Existing DP mechanisms (partial protection)

The attack code is modular and can be easily integrated into FeT training pipelines for robustness testing and defense evaluation.

