"""
Train FeT with Byzantine attacks.

This script extends train_fet.py to support Byzantine attacks during training.
Usage is identical to train_fet.py, with additional arguments for attack configuration.
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch.multiprocessing
import torch.nn as nn
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.VFLDataset import VFLSynAlignedDataset
from src.dataset.VFLRealDataset import VFLRealDataset
from src.preprocess.hdb.hdb_loader import load_both as load_both_hdb
from src.preprocess.ml_dataset.two_party_loader import TwoPartyLoader as FedSimSynLoader
from src.preprocess.nytaxi.ny_loader import NYBikeTaxiLoader
from src.train.Fit import fit
from src.utils.BasicUtils import (PartyPath, get_metric_from_str, get_metric_positive_from_str)
from src.utils.logger import CommLogger
from src.model.FeT import FeT
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy

# Avoid "Too many open files" error
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help="GPU ID. Set to None if you want to use CPU")

    # parameters for dataset
    parser.add_argument('--dataset', '-d', type=str,
                        help="dataset to use.")
    parser.add_argument('--n_parties', '-p', type=int, default=4,
                        help="number of parties. Should be >=2")
    parser.add_argument('--primary_party', '-pp', type=int, default=0,
                        help="primary party. Should be in [0, n_parties-1]")
    parser.add_argument('--splitter', '-sp', type=str, default='imp')
    parser.add_argument('--weights', '-w', type=float, default=1, help="weights for the ImportanceSplitter")
    parser.add_argument('--beta', '-b', type=float, default=1, help="beta for the CorrelationSplitter")
    parser.add_argument('--seed', '-s', type=int, default=0, help="random seed")
    parser.add_argument('--normalize_key', action='store_true', help="normalize key")
    parser.add_argument('--knn_k', type=int, default=1, help="k for kNN matching")
    parser.add_argument('--dp_sample', type=float, default=1.0, help="sample rate before topk")

    # parameters for model
    parser.add_argument('--data_embed_dim', type=int, default=128, help="data embedding dimension")
    parser.add_argument('--key_embed_dim', type=int, default=128, help="key embedding dimension")
    parser.add_argument('--num_heads', type=int, default=4, help="number of attention heads")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate")
    parser.add_argument('--party_dropout', type=float, default=0.0, help="party dropout rate")
    parser.add_argument('--n_local_blocks', type=int, default=1, help="number of local blocks")
    parser.add_argument('--n_agg_blocks', type=int, default=1, help="number of aggregation blocks")
    parser.add_argument('--activation', type=str, default='gelu', help="activation function")
    parser.add_argument('--disable_pe', action='store_true', help="disable positional encoding")
    parser.add_argument('--disable_dm', action='store_true', help="disable dynamic masking")

    # parameters for training
    parser.add_argument('--epochs', '-e', type=int, default=100, help="number of epochs")
    parser.add_argument('--batch_size', '-b', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="weight decay")
    parser.add_argument('--n_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--metric', type=str, default='acc', help="metric to use")
    parser.add_argument('--scheduler', type=str, default='plateau', help="scheduler type")
    parser.add_argument('--scheduler_patience', type=int, default=10, help="scheduler patience")
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help="scheduler factor")
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6, help="scheduler min lr")

    # parameters for differential privacy
    parser.add_argument('--dp_noise', type=float, default=None, help="DP noise scale")
    parser.add_argument('--dp_clip', type=float, default=None, help="DP clipping norm")

    # parameters for logging
    parser.add_argument('--log_dir', type=str, default='log', help="log directory")
    parser.add_argument('--model_path', type=str, default=None, help="path to save model")
    parser.add_argument('--visualize', action='store_true', help="visualize attention")
    parser.add_argument('--average_pe_freq', type=int, default=None, help="frequency to average PE")

    # ========== BYZANTINE ATTACK PARAMETERS ==========
    parser.add_argument('--byzantine_attack', action='store_true',
                        help="Enable Byzantine attacks")
    parser.add_argument('--attack_strategy', type=str, default='random_noise',
                        choices=['none', 'random_noise', 'sign_flip', 'scale_up', 
                                'zero_out', 'adversarial', 'gradient_scaling'],
                        help="Byzantine attack strategy")
    parser.add_argument('--attack_strength', type=float, default=1.0,
                        help="Attack strength (0.0 to 1.0 or higher)")
    parser.add_argument('--malicious_parties', type=str, default=None,
                        help="Comma-separated list of malicious party IDs (e.g., '1,2')")
    parser.add_argument('--attack_probability', type=float, default=1.0,
                        help="Probability of attack per batch (0.0 to 1.0)")
    # ==================================================

    args = parser.parse_args()

    # Parse malicious parties
    if args.malicious_parties:
        malicious_parties = [int(x.strip()) for x in args.malicious_parties.split(',')]
    else:
        malicious_parties = None  # All secondary parties

    # Set device
    if args.gpu is not None:
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    # Load dataset (same as train_fet.py)
    syn_root = "data/syn"
    real_root = "data/real"
    normalize_key = True
    
    # Dataset loading (simplified - see train_fet.py for full implementation)
    if args.dataset in ("gisette", "mnist"):
        normalize_key = False
        key_dim = 4
        syn_dataset_dir = f"data/syn/{args.dataset}/noise0.0/"  # Adjust as needed
        syn_aligned_dataset = VFLSynAlignedDataset.from_pickle(
            syn_dataset_dir, args.dataset, args.n_parties,
            primary_party_id=args.primary_party,
            splitter=args.splitter,
            weight=args.weights, beta=args.beta, seed=0,
            type=None
        )
        syn_dataset = VFLRealDataset.from_syn_aligned(
            syn_aligned_dataset, ks=args.knn_k, key_cols=key_dim,
            sample_rate_before_topk=args.dp_sample
        )
        train_dataset, val_dataset, test_dataset = syn_dataset.split_train_test_primary(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed
        )
    else:
        # Add other dataset loading logic here as needed
        raise ValueError(f"Dataset {args.dataset} not yet implemented in this script. "
                        f"Please use train_fet.py as reference or add dataset loading code.")
    
    # Normalize features
    X_scalers = train_dataset.normalize_(include_key=normalize_key)
    if val_dataset is not None:
        val_dataset.normalize_(scalers=X_scalers, include_key=normalize_key)
    test_dataset.normalize_(scalers=X_scalers, include_key=normalize_key)
    
    # Convert to tensors
    train_dataset.to_tensor_()
    test_dataset.to_tensor_()
    if val_dataset is not None:
        val_dataset.to_tensor_()
    
    # After dataset is loaded, create model
    if args.n_classes == 1:  # regression
        task = 'reg'
        loss_fn = nn.MSELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
    elif args.n_classes == 2:  # binary classification
        task = 'bin-cls'
        loss_fn = nn.BCELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
    else:  # multi-class classification
        task = 'multi-cls'
        loss_fn = nn.CrossEntropyLoss()
        out_dim = args.n_classes
        out_activation = None

    # Create base FeT model
    model = FeT(
        key_dims=[train_dataset.local_key_channels[i] for i in range(args.n_parties)],
        data_dims=[train_dataset.local_input_channels[i] for i in range(args.n_parties)],
        out_dim=out_dim,
        data_embed_dim=args.data_embed_dim,
        key_embed_dim=args.key_embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        party_dropout=args.party_dropout,
        n_embeddings=None,
        out_activation=out_activation,
        n_local_blocks=args.n_local_blocks,
        n_agg_blocks=args.n_agg_blocks,
        k=args.knn_k,
        rep_noise=args.dp_noise,
        max_rep_norm=args.dp_clip,
        enable_pe=not args.disable_pe,
        enable_dm=not args.disable_dm
    )

    # ========== BYZANTINE ATTACK SETUP ==========
    if args.byzantine_attack:
        print(f"\n{'='*60}")
        print("BYZANTINE ATTACK ENABLED")
        print(f"{'='*60}")
        print(f"Attack Strategy: {args.attack_strategy}")
        print(f"Attack Strength: {args.attack_strength}")
        print(f"Malicious Parties: {malicious_parties if malicious_parties else 'All secondary parties'}")
        print(f"Attack Probability: {args.attack_probability:.2%}")
        print(f"{'='*60}\n")
        
        # Create attacker
        strategy_map = {
            'none': AttackStrategy.NONE,
            'random_noise': AttackStrategy.RANDOM_NOISE,
            'sign_flip': AttackStrategy.SIGN_FLIP,
            'scale_up': AttackStrategy.SCALE_UP,
            'zero_out': AttackStrategy.ZERO_OUT,
            'adversarial': AttackStrategy.ADVERSARIAL,
            'gradient_scaling': AttackStrategy.GRADIENT_SCALING
        }
        
        attacker = ByzantineAttacker(
            strategy=strategy_map[args.attack_strategy],
            attack_strength=args.attack_strength,
            malicious_parties=malicious_parties,
            random_seed=args.seed
        )
        
        # Wrap model with attacker
        model = ByzantineFeT(
            fet_model=model,
            attacker=attacker,
            attack_probability=args.attack_probability
        )
    # =============================================

    # Create optimizer
    optimizer = optim.Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create scheduler
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.scheduler_factor,
            patience=args.scheduler_patience, min_lr=args.scheduler_min_lr
        )
    else:
        scheduler = None

    # Get metric function
    metric_fn = get_metric_from_str(args.metric)
    metric_positive = get_metric_positive_from_str(args.metric)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train
    fit(
        model, optimizer, loss_fn, metric_fn,
        train_loader, test_loader,
        epochs=args.epochs,
        gpu_id=args.gpu,
        n_classes=args.n_classes,
        task=task,
        scheduler=scheduler,
        has_key=True,
        val_loader=val_loader,
        metric_positive=metric_positive,
        y_scaler=None,
        writer=None,
        visualize=args.visualize,
        model_path=args.model_path,
        dataset_name=args.dataset,
        average_pe_freq=args.average_pe_freq
    )

    # Print attack statistics if Byzantine attack was used
    if args.byzantine_attack and hasattr(model, 'get_attack_stats'):
        stats = model.get_attack_stats()
        print(f"\n{'='*60}")
        print("BYZANTINE ATTACK STATISTICS")
        print(f"{'='*60}")
        print(f"Total attacks performed: {stats['attack_count']}")
        print(f"Total batches: {stats['total_batches']}")
        print(f"Attack rate: {stats['attack_rate']:.2%}")
        print(f"Strategy: {stats['strategy']}")
        print(f"{'='*60}\n")

