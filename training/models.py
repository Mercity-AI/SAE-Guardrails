"""
Neural network modules for SAE-based topic classification.
"""

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """Sparse autoencoder that keeps only the top-k activations per token (Qwen-Scope format)."""

    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        # W_enc stored as (d_sae, d_model) — Qwen convention; note Gemma Scope uses (d_model, d_sae)
        self.register_buffer("W_enc", torch.zeros(d_sae, d_model))
        self.register_buffer("b_enc", torch.zeros(d_sae))

    def fired_mask(self, x):
        """Return (n_tokens, d_sae) bool mask with True at each token's top-k feature positions."""
        # W_enc is (d_sae, d_model), so x @ W_enc.T gives (n_tokens, d_sae)
        pre = x @ self.W_enc.T + self.b_enc
        topk_idx = pre.topk(self.k, dim=-1).indices  # (n_tokens, k)
        mask = torch.zeros(x.shape[0], self.d_sae, dtype=torch.bool, device=x.device)
        mask.scatter_(1, topk_idx, True)
        return mask


class JumpReLUSAE(nn.Module):
    """JumpReLU sparse autoencoder as used in Google Gemma Scope (v1 .npz and v2 .safetensors)."""

    def __init__(self, d_model, d_sae):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        # W_enc is (d_model, d_sae) — Gemma Scope convention, transposed vs TopKSAE
        self.register_buffer("W_enc", torch.zeros(d_model, d_sae))
        self.register_buffer("b_enc", torch.zeros(d_sae))
        self.register_buffer("threshold", torch.zeros(d_sae))
        # b_dec is the pre-encoder bias (mean of the residual stream); must be subtracted
        # before encoding so that thresholds are calibrated against centred activations
        self.register_buffer("b_dec", torch.zeros(d_model))

    def fired_mask(self, x):
        """Return (n_tokens, d_sae) bool mask: True where pre-activation exceeds the JumpReLU threshold."""
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return pre > self.threshold


class TopicClassifier(nn.Module):
    """Two-layer MLP that maps SAE feature vectors to topic logits."""

    def __init__(self, n_in, n_classes, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)
