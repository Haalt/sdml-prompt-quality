import torch
from torch import nn

class PromptQualityModelV2(nn.Module):
    """
    ┌─ token path:  ID → Emb(d_t) → φ (2-layer shared MLP) → mean pool
    ├─ LoRA  path:  [ID Emb(d_L) ; weight] → 2-layer MLP → sum pool
    └─ concat [v_token, v_lora, cfg, n_loras, token_cnt] → MLP head → logit
    """

    def __init__(
        self,
        vocab_tokens: int,
        vocab_loras: int,
        d_t: int = 128,
        d_L: int = 64,
        d_token_hidden: int = 64,
        d_lora_hidden: int = 32,
        head_h1: int = 256,
        head_h2: int = 128,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        # embeddings
        self.token_embed = nn.Embedding(vocab_tokens, d_t, padding_idx=0)
        self.lora_embed = nn.Embedding(vocab_loras, d_L, padding_idx=0)

        # token φ (shared across tokens)
        self.token_phi = nn.Sequential(
            nn.Linear(d_t, d_token_hidden),
            nn.ReLU(),
            nn.Linear(d_token_hidden, d_t),  # back to d_t so we can pool
            nn.ReLU(),
        )

        # lora fusion ψ (shared across LoRAs)
        self.lora_fuse = nn.Sequential(
            nn.Linear(d_L + 1, d_lora_hidden),
            nn.ReLU(),
            nn.Linear(d_lora_hidden, d_L),
            nn.ReLU(),
        )

        # head
        in_dim = d_t + d_L + 2  # +3 scalars (cfg, n_loras, token_cnt)
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_h1),
            nn.BatchNorm1d(head_h1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_h1, head_h2),
            nn.BatchNorm1d(head_h2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_h2, 1),  # logits (no sigmoid)
        )

        # init
        nn.init.normal_(self.token_embed.weight, 0, 0.02)
        nn.init.normal_(self.lora_embed.weight, 0, 0.02)

    # ------------------------------------------------------------------ #
    def forward(self, tokens, token_mask, lora_ids, lora_w, cfg, n_loras):
        # token path
        t = self.token_embed(tokens)  # (B,T,d_t)
        t = self.token_phi(t)  # (B,T,d_t)
        t = t * token_mask.unsqueeze(-1)
        v_tokens = t.sum(1) / token_mask.sum(1, keepdim=True).clamp(min=1)  # (B,d_t)

        # LoRA path
        L_emb = self.lora_embed(lora_ids)  # (B,L,d_L)
        fused = torch.cat([L_emb, lora_w.unsqueeze(-1)], dim=-1)  # (B,L,d_L+1)
        fused = self.lora_fuse(fused)  # (B,L,d_L)
        real = (lora_ids != 0).unsqueeze(-1).float()
        v_loras = (fused * real).sum(1)  # (B,d_L)

        # concat all features
        z = torch.cat(
            [v_tokens, v_loras, cfg, n_loras], dim=1  # (B,1)  # (B,1)  # (B,1)
        )
        logits = self.head(z).squeeze(1)  # (B,)
        return logits