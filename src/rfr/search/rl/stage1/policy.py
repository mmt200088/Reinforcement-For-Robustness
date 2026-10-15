"""Stage-1 GTrXL actor-critic and return normalization."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from rfr.search.rl.stage1 import config as stage1_config

class RunningMeanStd:
    """Track return moments with Welford's online update."""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x_detached = x.detach()
            if not torch.is_floating_point(x_detached):
                x_detached = x_detached.float()
            batch_mean = float(x_detached.mean().item())
            batch_var = float(x_detached.var(unbiased=False).item())
            batch_count = int(x_detached.numel())
            self._update_from_moments(batch_mean, batch_var, batch_count)
            return
        x = np.asarray(x).flatten()

        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Welford's parallel algorithm for combining statistics"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count


        self.mean = self.mean + delta * batch_count / total_count


        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count

        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x):
        """Normalize values with the running mean and variance."""
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / (self.std + 1e-8)
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x):
        """Restore values from running normalization."""
        return x * self.std + self.mean


class GRUGate(nn.Module):
    """Gate a GTrXL residual branch with a trainable carry bias."""
    def __init__(self, d_model):
        super().__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Ur = nn.Linear(d_model, d_model, bias=False)
        self.Wz = nn.Linear(d_model, d_model, bias=False)
        self.Uz = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.bg = nn.Parameter(torch.ones(d_model) * 2.0)

    def forward(self, x, y):
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h_hat = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1 - z) * x + z * h_hat


class GTrXLBlock(nn.Module):
    """Pre-norm causal attention and feed-forward layers with GRU gates."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.gate1 = GRUGate(d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.gate2 = GRUGate(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = self.gate1(x, attn_out)

        norm_x2 = self.ln2(x)
        ff_out = self.ff(norm_x2)
        x = self.gate2(x, ff_out)
        return x


class GTrXLStrategyNetwork(nn.Module):
    """Shared causal GTrXL actor-critic used by the Stage-1 policy."""
    def __init__(self, num_layers=12, d_model=stage1_config.GTRXL_D_MODEL,
                 n_heads=stage1_config.GTRXL_N_HEADS, n_gtrxl_layers=stage1_config.GTRXL_N_LAYERS,
                 d_ff=stage1_config.GTRXL_D_FF, dropout=stage1_config.GTRXL_DROPOUT):
        super(GTrXLStrategyNetwork, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model


        self.embed_layer_idx = nn.Embedding(num_layers, stage1_config.POLICY_POSITION_DIM)
        self.embed_prev_g = nn.Embedding(stage1_config.SOS_TOKEN_GELU + 1, stage1_config.POLICY_ACTION_DIM)


        self.fc_continuous = nn.Sequential(
            nn.Linear(stage1_config.POLICY_CONTINUOUS_DIM, stage1_config.POLICY_PROJECTION_DIM),
            nn.LayerNorm(stage1_config.POLICY_PROJECTION_DIM),
            nn.SiLU()
        )


        token_input_dim = (
            stage1_config.POLICY_POSITION_DIM + stage1_config.POLICY_ACTION_DIM + stage1_config.POLICY_PROJECTION_DIM
        )
        self.input_proj = nn.Identity() if token_input_dim == d_model else nn.Linear(token_input_dim, d_model)


        self.gtrxl_blocks = nn.ModuleList([
            GTrXLBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_gtrxl_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)


        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh()
        )
        self.head_g = nn.Linear(64, 4)


        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self._initialize_weights()
        self._causal_mask_cache = {}

    def _initialize_weights(self):
        for module in [self.actor_head, self.critic_head, self.fc_continuous]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.head_g.weight, gain=0.01)
        nn.init.constant_(self.head_g.bias, 0.0)

        if isinstance(self.input_proj, nn.Linear):
            nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
            if self.input_proj.bias is not None:
                nn.init.constant_(self.input_proj.bias, 0.0)

        for block in self.gtrxl_blocks:
            for p in block.attn.in_proj_weight.chunk(3):
                nn.init.orthogonal_(p)
            nn.init.orthogonal_(block.attn.out_proj.weight)
            for layer in block.ff:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def _get_causal_mask(self, seq_len, device):
        """Return a cached upper-triangular causal mask."""
        if seq_len not in self._causal_mask_cache or self._causal_mask_cache[seq_len].device != device:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def _build_tokens(self, cont_features, layer_indices, prev_g_actions):
        emb_l = self.embed_layer_idx(layer_indices)
        emb_pg = self.embed_prev_g(prev_g_actions)
        feat_c = self.fc_continuous(cont_features)

        token_input = torch.cat([emb_l, emb_pg, feat_c], dim=-1)
        return self.input_proj(token_input)

    def forward(self, cont_features, layer_indices, prev_g_actions,
                gelu_mask=None, key_padding_mask=None):
        tokens = self._build_tokens(cont_features, layer_indices, prev_g_actions)
        seq_len = tokens.size(1)
        causal_mask = self._get_causal_mask(seq_len, tokens.device)

        x = tokens
        for block in self.gtrxl_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.ln_final(x)

        actor_feat = self.actor_head(x)
        logits_g = self.head_g(actor_feat)

        if gelu_mask is not None:
            logits_g = logits_g.masked_fill(~gelu_mask, float('-inf'))

        values = self.critic_head(x).squeeze(-1)

        return logits_g, values

    def get_action_and_logprob(self, cont_features, layer_indices, prev_g_actions,
                                return_probs=False, gelu_mask=None, key_padding_mask=None):
        logits_g, values = self.forward(
            cont_features, layer_indices, prev_g_actions,
            gelu_mask=gelu_mask, key_padding_mask=key_padding_mask
        )

        lg = logits_g[:, -1, :]
        val = values[:, -1]

        lg = lg.squeeze(0)
        val = val.squeeze(0)

        dist_g = Categorical(logits=lg)
        action_g = dist_g.sample()
        logprob = dist_g.log_prob(action_g)

        if return_probs:
            g_probs = torch.softmax(lg, dim=-1)
            return action_g, logprob, val, g_probs

        return action_g, logprob, val

    def evaluate_actions(self, cont_features, layer_indices, prev_g_actions,
                         actions_g, gelu_mask=None):
        logits_g, values = self.forward(
            cont_features, layer_indices, prev_g_actions,
            gelu_mask=gelu_mask
        )

        dist_g = Categorical(logits=logits_g)

        logprobs = dist_g.log_prob(actions_g)
        entropy = dist_g.entropy()

        return logprobs, entropy, values
