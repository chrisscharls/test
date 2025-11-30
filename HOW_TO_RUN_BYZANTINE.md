# How to Run Byzantine Attacks on FeT

## Quick Start

### Option 1: Simple Test Script (Quick Demo)

Run the test script to see a demonstration:

```bash
cd /home/chriss/FeT
python src/script/test_byzantine_attack.py
```

**Note**: This is a simplified demo that shows the attack structure but doesn't use real FeT datasets.

### Option 2: Full Integration with FeT Training (Recommended)

Use the integrated training script that works with real FeT datasets:

```bash
cd /home/chriss/FeT
python src/script/train_fet_byzantine.py \
    --dataset gisette \
    --n_parties 3 \
    --byzantine_attack \
    --attack_strategy random_noise \
    --attack_strength 2.0 \
    --malicious_parties "1,2" \
    --attack_probability 1.0 \
    --epochs 50 \
    --batch_size 128 \
    --gpu 0
```

### Option 3: Modify Existing train_fet.py

Add Byzantine attack support to your existing training script:

```python
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy

# After creating your FeT model:
if args.byzantine_attack:
    attacker = ByzantineAttacker(
        strategy=AttackStrategy.RANDOM_NOISE,
        attack_strength=2.0,
        malicious_parties=[1, 2]
    )
    model = ByzantineFeT(model, attacker, attack_probability=1.0)

# Then train normally
fit(model, optimizer, loss_fn, ...)
```

---

## Detailed Usage Examples

### Example 1: Random Noise Attack

```bash
python src/script/train_fet_byzantine.py \
    --dataset gisette \
    --n_parties 4 \
    --byzantine_attack \
    --attack_strategy random_noise \
    --attack_strength 1.5 \
    --malicious_parties "1,2" \
    --epochs 100 \
    --gpu 0
```

**What this does**:
- Trains FeT on gisette dataset with 4 parties
- Parties 1 and 2 are malicious
- They add random Gaussian noise to their representations
- Attack strength is 1.5x the standard deviation

### Example 2: Sign Flip Attack

```bash
python src/script/train_fet_byzantine.py \
    --dataset gisette \
    --n_parties 3 \
    --byzantine_attack \
    --attack_strategy sign_flip \
    --attack_strength 1.0 \
    --malicious_parties "1" \
    --attack_probability 0.5 \
    --epochs 100
```

**What this does**:
- Only party 1 is malicious
- Flips the sign of representations (reverses learning direction)
- Attacks 50% of batches (probabilistic)

### Example 3: Scale Up Attack

```bash
python src/script/train_fet_byzantine.py \
    --dataset gisette \
    --n_parties 5 \
    --byzantine_attack \
    --attack_strategy scale_up \
    --attack_strength 10.0 \
    --malicious_parties "2,3,4" \
    --epochs 100
```

**What this does**:
- Parties 2, 3, and 4 are malicious
- Scale their representations by 10x
- This makes them dominate the aggregation

### Example 4: Compare Normal vs. Attacked

**Normal Training:**
```bash
python src/script/train_fet.py \
    --dataset gisette \
    --n_parties 3 \
    --epochs 100 \
    --gpu 0
```

**Attacked Training:**
```bash
python src/script/train_fet_byzantine.py \
    --dataset gisette \
    --n_parties 3 \
    --byzantine_attack \
    --attack_strategy sign_flip \
    --attack_strength 1.0 \
    --malicious_parties "1,2" \
    --epochs 100 \
    --gpu 0
```

Compare the final accuracies to see the impact.

---

## Command Line Arguments

### Byzantine Attack Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--byzantine_attack` | flag | False | Enable Byzantine attacks |
| `--attack_strategy` | str | 'random_noise' | Attack type: 'random_noise', 'sign_flip', 'scale_up', 'zero_out', 'adversarial', 'gradient_scaling' |
| `--attack_strength` | float | 1.0 | Attack strength (0.0 to 1.0+). Higher = stronger attack |
| `--malicious_parties` | str | None | Comma-separated party IDs (e.g., "1,2"). None = all secondary parties |
| `--attack_probability` | float | 1.0 | Probability of attack per batch (0.0 to 1.0) |

### Standard FeT Arguments (same as train_fet.py)

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset name (e.g., 'gisette', 'mnist') |
| `--n_parties` | Number of parties |
| `--epochs` | Number of training epochs |
| `--batch_size` | Batch size |
| `--gpu` | GPU ID (or None for CPU) |
| `--lr` | Learning rate |
| ... | (all other train_fet.py arguments) |

---

## Python API Usage

### Basic Usage

```python
from src.model.FeT import FeT
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy

# 1. Create base FeT model
model = FeT(
    key_dims=[5, 5, 5],
    data_dims=[100, 150, 200],
    out_dim=1,
    data_embed_dim=128,
    # ... other parameters
)

# 2. Create attacker
attacker = ByzantineAttacker(
    strategy=AttackStrategy.RANDOM_NOISE,
    attack_strength=2.0,
    malicious_parties=[1, 2],  # Parties 1 and 2 are malicious
    random_seed=42
)

# 3. Wrap model
byzantine_model = ByzantineFeT(
    fet_model=model,
    attacker=attacker,
    attack_probability=1.0  # Attack every batch
)

# 4. Train normally
for epoch in range(epochs):
    for Xs, y in train_loader:
        y_pred = byzantine_model(Xs)  # Attacks injected automatically
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
```

### Different Attack Strategies

```python
# Random noise
attacker = ByzantineAttacker(
    strategy=AttackStrategy.RANDOM_NOISE,
    attack_strength=1.5
)

# Sign flip
attacker = ByzantineAttacker(
    strategy=AttackStrategy.SIGN_FLIP,
    attack_strength=1.0
)

# Scale up
attacker = ByzantineAttacker(
    strategy=AttackStrategy.SCALE_UP,
    attack_strength=10.0
)

# Zero out
attacker = ByzantineAttacker(
    strategy=AttackStrategy.ZERO_OUT,
    attack_strength=1.0  # Not used for zero out
)
```

### Get Attack Statistics

```python
# After training
stats = byzantine_model.get_attack_stats()
print(f"Attacks: {stats['attack_count']}/{stats['total_batches']}")
print(f"Rate: {stats['attack_rate']:.2%}")
```

---

## Integration with Existing Code

### Modify train_fet.py

Add this after model creation (around line 223):

```python
# Add import at top
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy

# Add argument parser option
parser.add_argument('--byzantine_attack', action='store_true')
parser.add_argument('--attack_strategy', type=str, default='random_noise')
parser.add_argument('--attack_strength', type=float, default=1.0)
parser.add_argument('--malicious_parties', type=str, default=None)

# After model creation, wrap if attack enabled
if args.byzantine_attack:
    strategy_map = {
        'random_noise': AttackStrategy.RANDOM_NOISE,
        'sign_flip': AttackStrategy.SIGN_FLIP,
        # ... etc
    }
    attacker = ByzantineAttacker(
        strategy=strategy_map[args.attack_strategy],
        attack_strength=args.attack_strength,
        malicious_parties=[int(x) for x in args.malicious_parties.split(',')] if args.malicious_parties else None
    )
    model = ByzantineFeT(model, attacker)
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src.attack'"

**Solution**: Make sure you're running from the project root:
```bash
cd /home/chriss/FeT
python src/script/train_fet_byzantine.py ...
```

### Issue: "Attack not working"

**Check**:
1. Is `--byzantine_attack` flag set?
2. Is `attack_probability > 0`?
3. Are malicious parties valid (not primary party)?
4. Check attack statistics: `model.get_attack_stats()`

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or use CPU:
```bash
--batch_size 64 --gpu None
```

---

## Expected Output

When running with Byzantine attacks, you should see:

```
============================================================
BYZANTINE ATTACK ENABLED
============================================================
Attack Strategy: random_noise
Attack Strength: 2.0
Malicious Parties: [1, 2]
Attack Probability: 100.00%
============================================================

Epoch: 1, Train Loss: 0.5234, Train Score: 0.65
Epoch: 2, Train Loss: 0.6123, Train Score: 0.58
...
(Note: Loss should be higher and accuracy lower than normal training)

============================================================
BYZANTINE ATTACK STATISTICS
============================================================
Total attacks performed: 1250
Total batches: 1250
Attack rate: 100.00%
Strategy: random_noise
============================================================
```

---

## Next Steps

1. **Test different attack strategies** to see their impact
2. **Vary attack strength** to find the breaking point
3. **Test with different numbers of malicious parties**
4. **Implement defense mechanisms** (robust aggregation, outlier detection)
5. **Compare performance** with and without attacks

For more details, see:
- `BYZANTINE_ATTACK_WORKFLOW.md` - Detailed workflow explanation
- `BYZANTINE_ATTACK_QUICK_START.md` - Quick reference guide

