"""
Byzantine Attack Strategies for FeT (Federated Transformer)

Byzantine attacks refer to malicious parties that send corrupted or adversarial
representations during the aggregation phase to disrupt model training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Callable
from enum import Enum


class AttackStrategy(Enum):
    """Enumeration of different Byzantine attack strategies"""
    NONE = "none"
    RANDOM_NOISE = "random_noise"
    SIGN_FLIP = "sign_flip"
    SCALE_UP = "scale_up"
    ZERO_OUT = "zero_out"
    ADVERSARIAL = "adversarial"
    LABEL_FLIP = "label_flip"
    GRADIENT_SCALING = "gradient_scaling"


class ByzantineAttacker:
    """
    Implements various Byzantine attack strategies for FeT.
    
    In FeT, Byzantine attacks occur when malicious secondary parties send
    corrupted representations during the cut layer aggregation phase.
    """
    
    def __init__(
        self,
        strategy: AttackStrategy = AttackStrategy.NONE,
        attack_strength: float = 1.0,
        malicious_parties: Optional[List[int]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Byzantine attacker.
        
        Args:
            strategy: Attack strategy to use
            attack_strength: Strength of the attack (0.0 to 1.0 or higher)
            malicious_parties: List of party IDs that are malicious (None = all secondary parties)
            random_seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.attack_strength = attack_strength
        self.malicious_parties = malicious_parties
        self.random_seed = random_seed
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
    
    def attack(
        self,
        representations: List[torch.Tensor],
        malicious_party_indices: List[int],
        primary_party_id: int = 0
    ) -> List[torch.Tensor]:
        """
        Apply Byzantine attack to representations.
        
        Args:
            representations: List of representations from secondary parties
            malicious_party_indices: Indices of malicious parties (relative to secondary parties)
            primary_party_id: ID of primary party (not attacked)
            
        Returns:
            Corrupted representations
        """
        if self.strategy == AttackStrategy.NONE:
            return representations
        
        corrupted_reps = [rep.clone() for rep in representations]
        
        for idx in malicious_party_indices:
            if idx < len(corrupted_reps):
                corrupted_reps[idx] = self._apply_strategy(corrupted_reps[idx])
        
        return corrupted_reps
    
    def _apply_strategy(self, rep: torch.Tensor) -> torch.Tensor:
        """Apply specific attack strategy to a representation."""
        if self.strategy == AttackStrategy.RANDOM_NOISE:
            return self._random_noise(rep)
        elif self.strategy == AttackStrategy.SIGN_FLIP:
            return self._sign_flip(rep)
        elif self.strategy == AttackStrategy.SCALE_UP:
            return self._scale_up(rep)
        elif self.strategy == AttackStrategy.ZERO_OUT:
            return self._zero_out(rep)
        elif self.strategy == AttackStrategy.ADVERSARIAL:
            return self._adversarial(rep)
        elif self.strategy == AttackStrategy.GRADIENT_SCALING:
            return self._gradient_scaling(rep)
        else:
            return rep
    
    def _random_noise(self, rep: torch.Tensor) -> torch.Tensor:
        """
        Add random Gaussian noise to representation.
        Attack: rep' = rep + N(0, attack_strength * std(rep))
        """
        noise_scale = self.attack_strength * rep.std().item()
        noise = torch.randn_like(rep) * noise_scale
        return rep + noise
    
    def _sign_flip(self, rep: torch.Tensor) -> torch.Tensor:
        """
        Flip the sign of the representation.
        Attack: rep' = -attack_strength * rep
        """
        return -self.attack_strength * rep
    
    def _scale_up(self, rep: torch.Tensor) -> torch.Tensor:
        """
        Scale up the representation to dominate aggregation.
        Attack: rep' = attack_strength * rep
        """
        return self.attack_strength * rep
    
    def _zero_out(self, rep: torch.Tensor) -> torch.Tensor:
        """
        Zero out the representation (dropout attack).
        Attack: rep' = 0
        """
        return torch.zeros_like(rep)
    
    def _adversarial(self, rep: torch.Tensor) -> torch.Tensor:
        """
        Create adversarial representation that maximizes disruption.
        Attack: rep' = rep + attack_strength * sign(gradient_estimate)
        """
        # Estimate adversarial direction (opposite of expected gradient)
        # In practice, this could use gradient information if available
        adversarial_direction = torch.sign(rep) * self.attack_strength
        return rep + adversarial_direction * rep.norm()
    
    def _gradient_scaling(self, rep: torch.Tensor) -> torch.Tensor:
        """
        Scale representation to create large gradient updates.
        Attack: rep' = attack_strength * rep (similar to scale_up but for gradients)
        """
        return self.attack_strength * rep


class ByzantineFeT(nn.Module):
    """
    Wrapper around FeT that allows Byzantine attacks during training.
    
    This class extends FeT to inject Byzantine attacks at the aggregation phase.
    """
    
    def __init__(
        self,
        fet_model: nn.Module,
        attacker: Optional[ByzantineAttacker] = None,
        attack_probability: float = 0.0
    ):
        """
        Initialize Byzantine FeT wrapper.
        
        Args:
            fet_model: Base FeT model
            attacker: Byzantine attacker instance
            attack_probability: Probability of attack per batch (0.0 to 1.0)
        """
        super().__init__()
        self.fet_model = fet_model
        self.attacker = attacker or ByzantineAttacker(strategy=AttackStrategy.NONE)
        self.attack_probability = attack_probability
        self.attack_count = 0
        self.total_batches = 0
    
    def forward(self, key_Xs, visualize=False, force_attack=False):
        """
        Forward pass with optional Byzantine attack.
        
        Args:
            key_Xs: Input key-data pairs
            visualize: Whether to visualize attention
            force_attack: Force attack regardless of probability
            
        Returns:
            Model output
        """
        # Monkey-patch the aggregation step to inject attack
        original_forward = self.fet_model.forward
        
        def attacked_forward(key_Xs, visualize=False):
            # Get secondary party representations before aggregation
            # We need to intercept at the cut layer
            
            # Call original forward but intercept at aggregation
            return self._forward_with_attack(key_Xs, visualize)
        
        # Temporarily replace forward method
        self.fet_model.forward = attacked_forward
        try:
            output = self.fet_model.forward(key_Xs, visualize)
        finally:
            # Restore original forward
            self.fet_model.forward = original_forward
        
        return output
    
    def _forward_with_attack(self, key_Xs, visualize=False):
        """
        Forward pass that intercepts aggregation for Byzantine attack.
        """
        # Check if we should attack this batch
        should_attack = (torch.rand(1).item() < self.attack_probability) or self.attacker.strategy != AttackStrategy.NONE
        
        if not should_attack:
            return self.fet_model.forward(key_Xs, visualize)
        
        # Extract representations before aggregation
        # This requires accessing internal FeT state
        return self._inject_attack_in_forward(key_Xs, visualize)
    
    def _inject_attack_in_forward(self, key_Xs, visualize=False):
        """
        Inject Byzantine attack during FeT forward pass.
        This method replicates FeT's forward logic but injects attack at aggregation.
        """
        # Access FeT's internal methods
        fet = self.fet_model
        
        # Validate inputs
        fet.check_args(key_Xs)
        
        # Prepare keys and data
        keys = []
        Xs = []
        for key_X in key_Xs:
            if len(key_X[0].shape) == 2:
                keys.append(key_X[0].unsqueeze(1))
                Xs.append(key_X[1].unsqueeze(1))
            else:
                keys.append(key_X[0])
                Xs.append(key_X[1])
        
        # Dynamic masking
        masks = []
        for i in range(fet.n_parties):
            if not fet.enable_dm:
                mask = None
            elif i == fet.primary_party_id:
                mask = None
            else:
                mask = fet.dynamic_mask_layers[i](keys[i].reshape(keys[i].shape[0], -1))
            masks.append(mask)
        
        # Embedding and positional encoding
        key_Xs = [torch.cat([keys[i], Xs[i]], dim=-1) for i in range(fet.n_parties)]
        key_X_embeds = []
        for i in range(fet.n_parties):
            key_X_embed = fet.data_embeddings[i](key_Xs[i])
            if fet.enable_pe:
                pe = fet.positional_encodings[i](keys[i].view(-1, 1, fet.key_dims[i]))
                key_X_embed += pe.view(key_X_embed.shape)
            key_X_embeds.append(key_X_embed)
        
        # Self-attention for secondary parties
        key_X_embeds = [
            fet.self_attns[i](key_X_embeds[i], need_weights=visualize, key_padding_mask=masks[i])
            if i != fet.primary_party_id else key_X_embeds[i]
            for i in range(fet.n_parties)
        ]
        
        # Extract secondary representations
        primary_key_X_embed = key_X_embeds[fet.primary_party_id]
        secondary_key_X_embeds = [
            key_X_embeds[i] for i in range(fet.n_parties) if i != fet.primary_party_id
        ]
        
        # Party dropout
        n_drop_parties = 0
        if fet.training and not np.isclose(fet.party_dropout, 0):
            n_drop_parties = int((fet.n_parties - 1) * fet.party_dropout)
            drop_party_indices = torch.randperm(fet.n_parties - 1)[:n_drop_parties]
            drop_mask = torch.ones(fet.n_parties - 1)
            drop_mask[drop_party_indices] = 0.
            drop_mask = drop_mask.to(primary_key_X_embed.device)
            secondary_key_X_embeds = [
                drop_mask[i] * secondary_key_X_embeds[i]
                for i in range(fet.n_parties - 1)
            ]
        
        # ========== BYZANTINE ATTACK INJECTION POINT ==========
        # This is where malicious parties corrupt their representations
        if self.attacker.strategy != AttackStrategy.NONE:
            # Determine which secondary parties are malicious
            n_secondary = len(secondary_key_X_embeds)
            if self.attacker.malicious_parties is None:
                # Attack all secondary parties
                malicious_indices = list(range(n_secondary))
            else:
                # Attack specified parties (map from global party ID to secondary index)
                malicious_indices = [
                    i for i, global_id in enumerate(
                        [j for j in range(fet.n_parties) if j != fet.primary_party_id]
                    ) if global_id in self.attacker.malicious_parties
                ]
            
            # Apply attack
            secondary_key_X_embeds = self.attacker.attack(
                secondary_key_X_embeds,
                malicious_indices,
                fet.primary_party_id
            )
            self.attack_count += 1
        self.total_batches += 1
        # =====================================================
        
        # Apply privacy mechanisms (DP noise and clipping)
        if fet.rep_noise is not None and fet.max_rep_norm is not None:
            max_rep_norm_per_party = fet.max_rep_norm / (fet.n_parties - 1)
            secondary_reps = []
            for secondary_key_X_embed in secondary_key_X_embeds:
                cut_layer_i = torch.tanh(secondary_key_X_embed)
                cut_layer_i_flat = cut_layer_i.reshape(cut_layer_i.shape[0], -1)
                per_sample_norm = torch.norm(cut_layer_i_flat, dim=1, p=2)
                clip_coef = max_rep_norm_per_party / (per_sample_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1)
                rep = cut_layer_i_flat * clip_coef_clamped.unsqueeze(-1)
                secondary_reps.append(rep)
            
            cut_layer_key_X_flat = torch.sum(torch.stack(secondary_reps), dim=0)
            noise = (torch.normal(0, fet.rep_noise * fet.max_rep_norm, cut_layer_key_X_flat.shape)
                    .to(cut_layer_key_X_flat.device))
            cut_layer_key_X_flat += noise
            cut_layer_key_X = cut_layer_key_X_flat.reshape(
                cut_layer_key_X_flat.shape[0], -1, fet.data_embed_dim
            )
        else:
            cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)
            cut_layer_key_X /= (fet.n_parties - n_drop_parties - 1)
        
        # Aggregation attention
        agg_key_X_embed = fet.agg_attn(
            primary_key_X_embed,
            cut_layer_key_X,
            need_weights=visualize,
            key_padding_mask=masks[fet.primary_party_id]
        )
        
        # Output layer
        output = fet.output_layer(agg_key_X_embed.reshape(agg_key_X_embed.shape[0], -1))
        if fet.out_activation is not None:
            output = fet.out_activation(output)
        
        return output
    
    def get_attack_stats(self):
        """Get statistics about attacks performed."""
        return {
            'attack_count': self.attack_count,
            'total_batches': self.total_batches,
            'attack_rate': self.attack_count / max(self.total_batches, 1),
            'strategy': self.attacker.strategy.value
        }


def create_byzantine_attacker(
    strategy: str = "random_noise",
    attack_strength: float = 1.0,
    malicious_parties: Optional[List[int]] = None,
    random_seed: Optional[int] = None
) -> ByzantineAttacker:
    """
    Factory function to create Byzantine attacker.
    
    Args:
        strategy: Attack strategy name
        attack_strength: Attack strength
        malicious_parties: List of malicious party IDs
        random_seed: Random seed
        
    Returns:
        ByzantineAttacker instance
    """
    strategy_enum = AttackStrategy[strategy.upper()] if hasattr(AttackStrategy, strategy.upper()) else AttackStrategy.NONE
    return ByzantineAttacker(
        strategy=strategy_enum,
        attack_strength=attack_strength,
        malicious_parties=malicious_parties,
        random_seed=random_seed
    )

