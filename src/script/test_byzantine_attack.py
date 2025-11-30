"""
Example script demonstrating Byzantine attacks on FeT model.

This script shows how to:
1. Create a FeT model
2. Wrap it with Byzantine attacker
3. Train with attacks
4. Compare normal vs. attacked performance
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.FeT import FeT
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy


class SimpleVFLDataset(Dataset):
    """
    Simple VFL dataset for testing.
    Returns data in format: [(key_0, data_0), (key_1, data_1), ...], y
    """
    def __init__(self, key_Xs, y):
        """
        Args:
            key_Xs: List of (key, data) tuples for each party
            y: Labels tensor
        """
        self.key_Xs = key_Xs  # List of tuples: [(key_0, data_0), (key_1, data_1), ...]
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Return [(key_0[idx], data_0[idx]), (key_1[idx], data_1[idx]), ...], y[idx]
        key_Xs_sample = [(key[idx], data[idx]) for key, data in self.key_Xs]
        return key_Xs_sample, self.y[idx]


def collate_fn(batch):
    """
    Custom collate function for VFL data.
    Batch is a list of (key_Xs, y) tuples.
    """
    key_Xs_list, y_list = zip(*batch)
    
    # key_Xs_list is a list of lists: [[(k0, d0), (k1, d1), ...], [(k0, d0), (k1, d1), ...], ...]
    # We need to batch each party's keys and data separately
    n_parties = len(key_Xs_list[0])
    
    # For each party, stack keys and data
    batched_key_Xs = []
    for party_idx in range(n_parties):
        # Get all (key, data) pairs for this party across the batch
        party_keys = torch.stack([key_Xs[party_idx][0] for key_Xs in key_Xs_list])
        party_data = torch.stack([key_Xs[party_idx][1] for key_Xs in key_Xs_list])
        batched_key_Xs.append((party_keys, party_data))
    
    # Stack labels
    batched_y = torch.stack(y_list)
    
    return batched_key_Xs, batched_y


def create_synthetic_data(n_samples=1000, n_parties=3, key_dim=5, data_dims=[100, 150, 200], n_classes=2):
    """Create synthetic multi-party data for testing."""
    key_Xs = []
    
    # Generate keys (same for all parties with small noise)
    base_keys = torch.randn(n_samples, key_dim)
    
    # Generate data for each party
    for i, data_dim in enumerate(data_dims):
        # Add noise to keys for each party
        keys = base_keys + 0.1 * torch.randn(n_samples, key_dim)
        # Generate features
        data = torch.randn(n_samples, data_dim)
        key_Xs.append((keys, data))
    
    # Generate labels (binary classification)
    y = torch.randint(0, n_classes, (n_samples,)).float()
    
    return key_Xs, y


def train_model(model, train_loader, epochs=10, device='cpu'):
    """Train a model and return loss history."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        for key_Xs, y in train_loader:
            # Format: key_Xs is a list of tuples, need to handle batching
            # Each key_Xs[i] is a tuple of (key_batch, data_batch) for party i
            # Move to device
            key_Xs_device = []
            for key, data in key_Xs:
                key_Xs_device.append((key.to(device), data.to(device)))
            y = y.to(device).float()
            
            optimizer.zero_grad()
            y_pred = model(key_Xs_device)
            y_pred = y_pred.flatten()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for key_Xs, y in test_loader:
            # Format: key_Xs is a list of tuples
            key_Xs_device = []
            for key, data in key_Xs:
                key_Xs_device.append((key.to(device), data.to(device)))
            y = y.to(device).float()
            
            y_pred = model(key_Xs_device).flatten()
            y_pred_binary = (y_pred > 0.5).float()
            correct += (y_pred_binary == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    """Main function to demonstrate Byzantine attacks."""
    print("=" * 60)
    print("Byzantine Attack Demonstration on FeT")
    print("=" * 60)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    key_Xs, y = create_synthetic_data(n_samples=1000, n_parties=3)
    
    # Split into train/test
    train_size = int(0.8 * len(y))
    train_key_Xs = [(k[:train_size], d[:train_size]) for k, d in key_Xs]
    train_y = y[:train_size]
    test_key_Xs = [(k[train_size:], d[train_size:]) for k, d in key_Xs]
    test_y = y[train_size:]
    
    # Create data loaders using custom dataset
    train_dataset = SimpleVFLDataset(train_key_Xs, train_y)
    test_dataset = SimpleVFLDataset(test_key_Xs, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Model parameters
    key_dims = [5, 5, 5]
    data_dims = [100, 150, 200]
    
    print("\n2. Creating FeT model...")
    # Create normal model
    normal_model = FeT(
        key_dims=key_dims,
        data_dims=data_dims,
        out_dim=1,
        data_embed_dim=64,
        num_heads=2,
        dropout=0.1,
        n_local_blocks=1,
        n_agg_blocks=1,
        primary_party_id=0,
        enable_pe=True,
        enable_dm=True
    )
    
    print("Model created successfully!")
    print(f"  - Key dimensions: {key_dims}")
    print(f"  - Data dimensions: {data_dims}")
    print(f"  - Number of parties: {len(key_dims)}")
    
    # Demonstrate attack strategies
    print("\n3. Demonstrating different attack strategies...")
    
    strategies = [
        (AttackStrategy.RANDOM_NOISE, "Random Noise Attack"),
        (AttackStrategy.SIGN_FLIP, "Sign Flip Attack"),
        (AttackStrategy.SCALE_UP, "Scale Up Attack"),
        (AttackStrategy.ZERO_OUT, "Zero Out Attack"),
    ]
    
    for strategy, name in strategies:
        print(f"\n   {name}:")
        attacker = ByzantineAttacker(
            strategy=strategy,
            attack_strength=2.0,
            malicious_parties=[1, 2],  # Secondary parties 1 and 2
            random_seed=42
        )
        
        # Create attacked model (create new instance to avoid state sharing)
        base_model = FeT(
            key_dims=key_dims,
            data_dims=data_dims,
            out_dim=1,
            data_embed_dim=64,
            num_heads=2,
            dropout=0.1,
            n_local_blocks=1,
            n_agg_blocks=1,
            primary_party_id=0,
            enable_pe=True,
            enable_dm=True
        )
        
        attacked_model = ByzantineFeT(
            fet_model=base_model,
            attacker=attacker,
            attack_probability=1.0
        )
        
        print(f"   - Strategy: {strategy.value}")
        print(f"   - Attack strength: 2.0")
        print(f"   - Malicious parties: [1, 2]")
        print(f"   - Attack probability: 100%")
        
        # Test a single forward pass
        try:
            sample_batch = next(iter(train_loader))
            key_Xs_sample, y_sample = sample_batch
            key_Xs_device = [(k.to('cpu'), d.to('cpu')) for k, d in key_Xs_sample]
            with torch.no_grad():
                output = attacked_model(key_Xs_device)
            print(f"   - Forward pass successful! Output shape: {output.shape}")
        except Exception as e:
            print(f"   - Forward pass failed: {e}")
    
    print("\n4. Testing Attack with Training:")
    print("   Training a small model with Byzantine attack...")
    
    # Create a fresh model for training
    train_model_instance = FeT(
        key_dims=key_dims,
        data_dims=data_dims,
        out_dim=1,
        data_embed_dim=32,  # Smaller for faster demo
        num_heads=2,
        dropout=0.1,
        n_local_blocks=1,
        n_agg_blocks=1,
        primary_party_id=0,
        enable_pe=False,  # Disable for faster demo
        enable_dm=False  # Disable for faster demo
    )
    
    attacker = ByzantineAttacker(
        strategy=AttackStrategy.RANDOM_NOISE,
        attack_strength=1.5,
        malicious_parties=[1, 2]
    )
    
    attacked_model = ByzantineFeT(
        fet_model=train_model_instance,
        attacker=attacker,
        attack_probability=1.0
    )
    
    # Train for a few epochs
    print("   Training for 3 epochs...")
    losses = train_model(attacked_model, train_loader, epochs=3, device='cpu')
    
    # Get attack statistics
    stats = attacked_model.get_attack_stats()
    print(f"\n   Attack Statistics:")
    print(f"   - Attacks performed: {stats['attack_count']}")
    print(f"   - Total batches: {stats['total_batches']}")
    print(f"   - Attack rate: {stats['attack_rate']:.2%}")
    print(f"   - Strategy: {stats['strategy']}")
    
    print("\n" + "=" * 60)
    print("Byzantine Attack Demonstration Complete!")
    print("=" * 60)
    print("\nTo use in practice:")
    print("1. Create your FeT model")
    print("2. Create ByzantineAttacker with desired strategy")
    print("3. Wrap model with ByzantineFeT")
    print("4. Train normally - attacks are injected automatically")
    print("\nSee BYZANTINE_ATTACK_WORKFLOW.md for detailed workflow.")


if __name__ == "__main__":
    main()

