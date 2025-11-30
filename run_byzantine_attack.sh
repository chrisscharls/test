#!/bin/bash
# Simple script to run FeT with Byzantine attacks
# Usage: ./run_byzantine_attack.sh [dataset] [attack_strategy] [attack_strength]

DATASET=${1:-gisette}
ATTACK_STRATEGY=${2:-random_noise}
ATTACK_STRENGTH=${3:-2.0}
N_PARTIES=${4:-3}
MALICIOUS_PARTIES=${5:-"1,2"}

echo "=========================================="
echo "Running FeT with Byzantine Attack"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Attack Strategy: $ATTACK_STRATEGY"
echo "Attack Strength: $ATTACK_STRENGTH"
echo "Number of Parties: $N_PARTIES"
echo "Malicious Parties: $MALICIOUS_PARTIES"
echo "=========================================="

# First, let's create a simple Python script that wraps the attack
python3 << EOF
import sys
import os
sys.path.append('.')

from src.model.FeT import FeT
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy
from src.script.train_fet import *  # Import everything from train_fet

# Parse arguments (simplified)
class Args:
    def __init__(self):
        self.dataset = "$DATASET"
        self.n_parties = $N_PARTIES
        self.primary_party = 0
        self.splitter = 'imp'
        self.weights = 1.0
        self.beta = 1.0
        self.seed = 0
        self.key_noise = 0.0
        self.knn_k = 100
        self.dp_sample = None
        self.disable_cache = True
        self.flush_cache = False
        self.epochs = 50
        self.batch_size = 128
        self.lr = 0.001
        self.weight_decay = 0.0001
        self.n_classes = 2
        self.metric = 'acc'
        self.data_embed_dim = 128
        self.key_embed_dim = 128
        self.num_heads = 4
        self.dropout = 0.1
        self.party_dropout = 0.0
        self.n_local_blocks = 1
        self.n_agg_blocks = 1
        self.disable_pe = False
        self.disable_dm = False
        self.pe_average_freq = 0
        self.dp_noise = None
        self.dp_clip = None
        self.gpu = 0
        self.log_dir = 'log'
        self.result_path = None
        self.visualize = False
        # Byzantine attack parameters
        self.byzantine_attack = True
        self.attack_strategy = "$ATTACK_STRATEGY"
        self.attack_strength = $ATTACK_STRENGTH
        self.malicious_parties = "$MALICIOUS_PARTIES"
        self.attack_probability = 1.0

# Use the main logic from train_fet.py but add Byzantine attack
# This is a simplified version - for full functionality, modify train_fet.py directly
print("Note: For full functionality, modify train_fet.py to add Byzantine attack support.")
print("See HOW_TO_RUN_BYZANTINE.md for instructions.")
EOF

echo ""
echo "For a complete solution, see HOW_TO_RUN_BYZANTINE.md"
echo "Or modify train_fet.py directly as shown in the documentation."

