# Quick Run Guide - Byzantine Attacks on FeT

## Simplest Way to Run

### Step 1: Modify train_fet.py (One-time setup)

Add these imports at the top of `src/script/train_fet.py`:

```python
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy
```

Add these arguments to the argument parser (around line 88):

```python
# Byzantine attack parameters
parser.add_argument('--byzantine_attack', action='store_true',
                    help="Enable Byzantine attacks")
parser.add_argument('--attack_strategy', type=str, default='random_noise',
                    choices=['none', 'random_noise', 'sign_flip', 'scale_up', 
                            'zero_out', 'adversarial', 'gradient_scaling'],
                    help="Byzantine attack strategy")
parser.add_argument('--attack_strength', type=float, default=1.0,
                    help="Attack strength")
parser.add_argument('--malicious_parties', type=str, default=None,
                    help="Comma-separated list of malicious party IDs")
parser.add_argument('--attack_probability', type=float, default=1.0,
                    help="Probability of attack per batch")
```

Add this code right after model creation (around line 223, after `model = FeT(...)`):

```python
# ========== BYZANTINE ATTACK ==========
if args.byzantine_attack:
    strategy_map = {
        'none': AttackStrategy.NONE,
        'random_noise': AttackStrategy.RANDOM_NOISE,
        'sign_flip': AttackStrategy.SIGN_FLIP,
        'scale_up': AttackStrategy.SCALE_UP,
        'zero_out': AttackStrategy.ZERO_OUT,
        'adversarial': AttackStrategy.ADVERSARIAL,
        'gradient_scaling': AttackStrategy.GRADIENT_SCALING
    }
    
    malicious_parties = None
    if args.malicious_parties:
        malicious_parties = [int(x.strip()) for x in args.malicious_parties.split(',')]
    
    attacker = ByzantineAttacker(
        strategy=strategy_map[args.attack_strategy],
        attack_strength=args.attack_strength,
        malicious_parties=malicious_parties,
        random_seed=args.seed
    )
    
    model = ByzantineFeT(
        fet_model=model,
        attacker=attacker,
        attack_probability=args.attack_probability
    )
    
    print(f"\n{'='*60}")
    print("BYZANTINE ATTACK ENABLED")
    print(f"{'='*60}")
    print(f"Strategy: {args.attack_strategy}")
    print(f"Strength: {args.attack_strength}")
    print(f"Malicious Parties: {malicious_parties if malicious_parties else 'All secondary'}")
    print(f"Probability: {args.attack_probability:.2%}")
    print(f"{'='*60}\n")
# ======================================
```

### Step 2: Run with Attacks

```bash
# Normal training
python src/script/train_fet.py \
    --dataset gisette \
    --n_parties 3 \
    --epochs 50 \
    --gpu 0

# With Byzantine attack
python src/script/train_fet.py \
    --dataset gisette \
    --n_parties 3 \
    --epochs 50 \
    --gpu 0 \
    --byzantine_attack \
    --attack_strategy random_noise \
    --attack_strength 2.0 \
    --malicious_parties "1,2"
```

---

## Alternative: Use Python API Directly

Create a simple script `run_attack.py`:

```python
import sys
import os
sys.path.append('.')

from src.model.FeT import FeT
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy
from src.script.train_fet import *  # Import train_fet logic

# Your existing train_fet.py code here...
# But wrap the model with ByzantineFeT before training

# After creating model:
if True:  # Enable attack
    attacker = ByzantineAttacker(
        strategy=AttackStrategy.RANDOM_NOISE,
        attack_strength=2.0,
        malicious_parties=[1, 2]
    )
    model = ByzantineFeT(model, attacker, attack_probability=1.0)

# Continue with normal training...
```

---

## Test Script (Simplified Demo)

Run the test script to see the structure:

```bash
python src/script/test_byzantine_attack.py
```

This shows how attacks work but uses synthetic data.

---

## Full Examples

### Example 1: Random Noise Attack

```bash
python src/script/train_fet.py \
    --dataset gisette \
    --n_parties 4 \
    --byzantine_attack \
    --attack_strategy random_noise \
    --attack_strength 1.5 \
    --malicious_parties "1,2" \
    --epochs 100 \
    --gpu 0
```

### Example 2: Sign Flip Attack

```bash
python src/script/train_fet.py \
    --dataset gisette \
    --n_parties 3 \
    --byzantine_attack \
    --attack_strategy sign_flip \
    --attack_strength 1.0 \
    --malicious_parties "1" \
    --attack_probability 0.5 \
    --epochs 100
```

### Example 3: Scale Up Attack

```bash
python src/script/train_fet.py \
    --dataset gisette \
    --n_parties 5 \
    --byzantine_attack \
    --attack_strategy scale_up \
    --attack_strength 10.0 \
    --malicious_parties "2,3,4" \
    --epochs 100
```

---

## Verify Attack is Working

After training, check the output. You should see:

1. **Attack enabled message** at the start
2. **Higher training loss** compared to normal training
3. **Lower accuracy** compared to normal training
4. **Attack statistics** (if you add code to print them)

Add this at the end of training to see stats:

```python
if args.byzantine_attack and hasattr(model, 'get_attack_stats'):
    stats = model.get_attack_stats()
    print(f"\nAttack Stats: {stats}")
```

---

## Troubleshooting

**Q: Attack not working?**
- Check `--byzantine_attack` flag is set
- Verify `attack_probability > 0`
- Ensure malicious parties are valid (not primary party)

**Q: Import error?**
- Make sure you're in the project root: `cd /home/chriss/FeT`
- Check `src/attack/__init__.py` exists

**Q: Model not training?**
- Attacks should not prevent training, just degrade performance
- Check if loss is increasing (attack might be too strong)

---

## Next Steps

1. **Compare results**: Run with and without attacks
2. **Vary attack strength**: Find the breaking point
3. **Test different strategies**: See which is most effective
4. **Implement defenses**: Add robust aggregation, outlier detection

For more details, see:
- `HOW_TO_RUN_BYZANTINE.md` - Detailed instructions
- `BYZANTINE_ATTACK_WORKFLOW.md` - Complete workflow explanation

