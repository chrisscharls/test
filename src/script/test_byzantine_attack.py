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
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.FeT import FeT
from src.attack import ByzantineAttacker, ByzantineFeT, AttackStrategy


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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for Xs, y in train_loader:
            # Move to device
            Xs = [(Xi[0].to(device), Xi[1].to(device)) for Xi in Xs]
            y = y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(Xs)
            y_pred = y_pred.flatten()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for Xs, y in test_loader:
            Xs = [(Xi[0].to(device), Xi[1].to(device)) for Xi in Xs]
            y = y.to(device)
            
            y_pred = model(Xs).flatten()
            y_pred_binary = (y_pred > 0.5).float()
            correct += (y_pred_binary == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
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
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.stack([torch.cat([k, d], dim=-1) for k, d in train_key_Xs], dim=0),
        train_y
    )
    # Note: This is simplified - in practice, you'd use proper VFL dataset
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(
            torch.stack([torch.cat([k, d], dim=-1) for k, d in test_key_Xs], dim=0),
            test_y
        ),
        batch_size=32
    )
    
    # Model parameters
    key_dims = [5, 5, 5]
    data_dims = [100, 150, 200]
    
    print("\n2. Training normal FeT model...")
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
    
    # Note: This is a simplified example
    # In practice, you'd need to properly format the data for FeT
    print("Note: This is a simplified example. Full integration requires proper VFL dataset format.")
    
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
        
        # Create attacked model
        attacked_model = ByzantineFeT(
            fet_model=normal_model,
            attacker=attacker,
            attack_probability=1.0
        )
        
        print(f"   - Strategy: {strategy.value}")
        print(f"   - Attack strength: 2.0")
        print(f"   - Malicious parties: [1, 2]")
        print(f"   - Attack probability: 100%")
    
    print("\n4. Attack Statistics Example:")
    attacker = ByzantineAttacker(
        strategy=AttackStrategy.RANDOM_NOISE,
        attack_strength=1.5
    )
    attacked_model = ByzantineFeT(normal_model, attacker, attack_probability=0.5)
    
    # Simulate some forward passes
    for _ in range(10):
        # In real usage, this would be actual forward passes
        attacked_model.total_batches += 1
        if np.random.rand() < 0.5:
            attacked_model.attack_count += 1
    
    stats = attacked_model.get_attack_stats()
    print(f"   - Attacks performed: {stats['attack_count']}")
    print(f"   - Total batches: {stats['total_batches']}")
    print(f"   - Attack rate: {stats['attack_rate']:.2%}")
    
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

