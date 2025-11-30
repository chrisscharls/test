# Byzantine Attacks on FeT - Quick Start Guide

## Quick Overview

Byzantine attacks occur when **malicious secondary parties** send corrupted representations during the aggregation phase in FeT.

## Attack Location

```
FeT Forward Pass:
  ├─► Local Processing (All Parties)
  ├─► Extract Representations
  │
  ├─► ⚠️ BYZANTINE ATTACK INJECTION POINT ⚠️
  │   └─► Malicious parties corrupt their representations
  │
  ├─► Aggregation (sum of representations)
  ├─► Primary Party Attention
  └─► Output
```

## Basic Usage

### 1. Import Required Modules

```python
from src.model.FeT import FeT
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy
```

### 2. Create Base Model

```python
model = FeT(
    key_dims=[5, 5, 5],
    data_dims=[100, 150, 200],
    out_dim=1,
    data_embed_dim=128,
    # ... other parameters
)
```

### 3. Create Attacker

```python
attacker = ByzantineAttacker(
    strategy=AttackStrategy.RANDOM_NOISE,  # Attack type
    attack_strength=2.0,                     # How strong
    malicious_parties=[1, 2],                # Which parties are malicious
    random_seed=42
)
```

### 4. Wrap Model

```python
byzantine_model = ByzantineFeT(
    fet_model=model,
    attacker=attacker,
    attack_probability=1.0  # Attack every batch (or 0.5 for 50%)
)
```

### 5. Train Normally

```python
for epoch in range(epochs):
    for Xs, y in train_loader:
        y_pred = byzantine_model(Xs)  # Attacks injected automatically
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
```

## Available Attack Strategies

| Strategy | Description | Effect |
|----------|-------------|--------|
| `RANDOM_NOISE` | Add Gaussian noise | Degrades accuracy |
| `SIGN_FLIP` | Flip sign of representation | Reverses learning |
| `SCALE_UP` | Scale up representation | Dominates aggregation |
| `ZERO_OUT` | Send zero representation | Removes contribution |
| `ADVERSARIAL` | Maximize disruption | Training instability |

## Attack Workflow

```
Step 1: Normal Processing
  └─► All parties process data through self-attention

Step 2: Extract Representations
  └─► Get representations from secondary parties

Step 3: ⚠️ ATTACK INJECTION
  └─► Malicious parties corrupt their representations
      ├─► Random Noise: rep + noise
      ├─► Sign Flip: -rep
      ├─► Scale Up: 10x * rep
      └─► etc.

Step 4: Aggregation
  └─► Sum corrupted + honest representations

Step 5: Primary Party Processing
  └─► Primary attends to corrupted aggregate

Step 6: Output
  └─► Model produces corrupted predictions
```

## Example: Compare Normal vs. Attacked

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

# Visualize impact
import matplotlib.pyplot as plt
plt.plot(normal_losses, label='Normal')
plt.plot(attacked_losses, label='Byzantine Attack')
plt.legend()
plt.show()
```

## Attack Statistics

```python
# After training, check attack stats
stats = byzantine_model.get_attack_stats()
print(f"Attacks: {stats['attack_count']}/{stats['total_batches']}")
print(f"Rate: {stats['attack_rate']:.2%}")
```

## Files Created

1. **`src/attack/ByzantineAttack.py`** - Attack implementation
2. **`src/attack/__init__.py`** - Module exports
3. **`BYZANTINE_ATTACK_WORKFLOW.md`** - Detailed workflow
4. **`src/script/test_byzantine_attack.py`** - Example script

## Key Points

- ✅ Attacks are injected **automatically** during forward pass
- ✅ Attacks occur at **aggregation phase** (cut layer)
- ✅ Can target **specific parties** or all secondary parties
- ✅ Multiple **attack strategies** available
- ✅ **Probabilistic attacks** supported (attack_probability)

## Next Steps

1. Read `BYZANTINE_ATTACK_WORKFLOW.md` for detailed explanation
2. Run `test_byzantine_attack.py` to see examples
3. Integrate into your training pipeline
4. Test defense mechanisms

