import os

import torch
from torch import nn


class PromptQualityModelV3(nn.Module):
    """
    Prompt-quality scorer that extends V2 with sampler/step and upscaler-aware branches.

    Interface (all tensors are batch-first):
        tokens         : (B, T)    int64
        token_mask     : (B, T)    float32  {0,1}
        lora_ids       : (B, L)    int64
        lora_w         : (B, L)    float32  [0-1]
        cfg            : (B,)      float32  [0-1]
        n_loras        : (B,)      float32  [0-1]

        sampler_id     : (B,)      int64
        steps_log      : (B,)      float32  [0-1]  (log-scaled)
        steps_bucket   : (B,)      int64    0-(bucket_size-1)

        upscaler_id    : (B,)      int64
        up_has         : (B,)      float32  {0,1}
        up_steps       : (B,)      float32  [0-1]
        denoise        : (B,)      float32  [0-1]

    Output:
        logits         : (B,)      float32   (apply sigmoid in prod)
    """

    def __init__(
        self,
        vocab_tokens: int,
        vocab_loras: int,
        vocab_samplers: int,
        vocab_upscalers: int,
        bucket_size: int = 10,
        # dimensions
        d_t: int = 128,
        d_L: int = 64,
        d_s: int = 32,
        d_u: int = 32,
        d_token_hidden: int = 64,
        d_lora_hidden: int = 32,
        d_samp_hidden: int = 32,
        d_up_hidden: int = 32,
        d_bucket_emb: int = 8,
        head_h1: int = 320,
        head_h2: int = 160,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        # embeddings
        self.token_embed = nn.Embedding(vocab_tokens, d_t, padding_idx=0)
        self.lora_embed = nn.Embedding(vocab_loras, d_L, padding_idx=0)
        self.sampler_embed = nn.Embedding(vocab_samplers, d_s, padding_idx=0)
        self.bucket_embed = nn.Embedding(bucket_size, d_bucket_emb)
        self.upsc_embed = nn.Embedding(vocab_upscalers, d_u, padding_idx=0)

        # token φ
        self.token_phi = nn.Sequential(
            nn.Linear(d_t, d_token_hidden),
            nn.ReLU(),
            nn.Linear(d_token_hidden, d_t),
            nn.ReLU(),
        )

        # LoRA fusion ψL
        self.lora_fuse = nn.Sequential(
            nn.Linear(d_L + 1, d_lora_hidden),
            nn.ReLU(),
            nn.Linear(d_lora_hidden, d_L),
            nn.ReLU(),
        )

        # sampler/steps fusion ψS
        samp_in = d_s + d_bucket_emb + 1  # + steps_log
        self.samp_fuse = nn.Sequential(
            nn.Linear(samp_in, d_samp_hidden),
            nn.ReLU(),
            nn.Linear(d_samp_hidden, d_s),
            nn.ReLU(),
        )

        # upscaler fusion ψU
        up_in = d_u + 3  # up_has, up_steps, denoise
        self.up_fuse = nn.Sequential(
            nn.Linear(up_in, d_up_hidden),
            nn.ReLU(),
            nn.Linear(d_up_hidden, d_u),
            nn.ReLU(),
        )

        # head
        in_dim = d_t + d_L + d_s + d_u + 2  # scalars: cfg, n_loras
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_h1),
            nn.BatchNorm1d(head_h1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_h1, head_h2),
            nn.BatchNorm1d(head_h2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_h2, 1),
        )

        # init
        for emb in (
            self.token_embed,
            self.lora_embed,
            self.sampler_embed,
            self.upsc_embed,
            self.bucket_embed,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self,
                tokens, token_mask,
                lora_ids, lora_w,
                cfg, n_loras,
                sampler_id, steps_log, steps_bucket,
                upscaler_id, up_has, up_steps, denoise):

        B = tokens.size(0)

        # helper
        def to_col(x, B):
            """ensure tensor is (B,1). accepts (B,) or (B,1)"""
            return x.view(B, 1) if x.dim() == 1 else x

        # token path
        t = self.token_embed(tokens)
        t = self.token_phi(t)
        t = t * token_mask.unsqueeze(-1)
        v_tokens = t.sum(1) / token_mask.sum(1, keepdim=True).clamp(min=1)

        # LoRA path
        L_emb = self.lora_embed(lora_ids)
        fused = torch.cat([L_emb, lora_w.unsqueeze(-1)], -1)
        fused = self.lora_fuse(fused)
        real = (lora_ids != 0).unsqueeze(-1).float()
        v_loras = (fused * real).sum(1)

        # sampler / steps path
        sampler_id = sampler_id.squeeze(-1)
        steps_bucket = steps_bucket.squeeze(-1)

        samp_emb = self.sampler_embed(sampler_id)  # (B,d_s)
        bucket_emb = self.bucket_embed(steps_bucket)  # (B,d_b)
        steps_log = to_col(steps_log, B)  # (B,1)

        samp_cat = torch.cat([samp_emb, steps_log, bucket_emb], dim=-1)
        v_samp = self.samp_fuse(samp_cat)  # (B,d_s)

        # upscaler path
        upscaler_id = upscaler_id.squeeze(-1)
        up_emb = self.upsc_embed(upscaler_id)  # (B,d_u)

        up_has = to_col(up_has,   B)
        up_steps = to_col(up_steps, B)
        denoise = to_col(denoise,  B)

        up_cat = torch.cat([up_emb, up_has, up_steps, denoise], dim=-1)
        v_up = self.up_fuse(up_cat)  # (B,d_u)

        # scalar cols
        cfg_col = to_col(cfg, B)
        n_loras_col = to_col(n_loras, B)

        # concat & head
        z = torch.cat(
            [v_tokens, v_loras, v_samp, v_up,
             cfg_col, n_loras_col],
            dim=1
        )                                              # (B, in_dim)
        logits = self.head(z).squeeze(1)
        return logits


def load_model(checkpoint_path):
    torch.serialization.safe_globals(
        [PromptQualityModelV3])

   # tokenizer
#    try:
#         tokenizer = Tokenizer.load_from_file(TOKENIZER_FILE)
#     except Exception as e:
#         raise RuntimeError(f"Could not load tokenizer: {e}")

    vocab_tokens = 1139
    vocab_loras = 143
    vocab_samplers = 20
    vocab_upscalers = 17

    max_cfg_scale = 11.0
    max_loras = 7

    # instantiate & load
    model = PromptQualityModelV3(
        vocab_tokens=vocab_tokens,
        vocab_loras=vocab_loras,
        vocab_samplers=vocab_samplers,
        vocab_upscalers=vocab_upscalers,
        d_t=256,
        d_L=128,
        head_h1=512,
        dropout_p=0.4,
    )

    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    model.eval()

    return model
