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
        vocab_models: int = 2,
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

        d_m = 32
        self.d_m = d_m

        # embeddings
        self.token_embed = nn.Embedding(vocab_tokens, d_t, padding_idx=0)
        self.lora_embed = nn.Embedding(vocab_loras, d_L, padding_idx=0)
        self.sampler_embed = nn.Embedding(vocab_samplers, d_s, padding_idx=0)
        self.bucket_embed = nn.Embedding(bucket_size, d_bucket_emb)
        self.upsc_embed = nn.Embedding(vocab_upscalers, d_u, padding_idx=0)
        self.model_domain_embed = nn.Embedding(vocab_models, d_m)

        # token φ
        self.model_token_proj = nn.Linear(d_m, d_t)
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
        samp_in = d_s + d_bucket_emb + 1 + d_m  # + steps_log + model_id
        self.samp_fuse = nn.Sequential(
            nn.Linear(samp_in, d_samp_hidden),
            nn.ReLU(),
            nn.Linear(d_samp_hidden, d_s),
            nn.ReLU(),
        )

        # upscaler fusion ψU
        up_in = d_u + 3 + d_m  # up_has, up_steps, denoise, model_id
        self.up_fuse = nn.Sequential(
            nn.Linear(up_in, d_up_hidden),
            nn.ReLU(),
            nn.Linear(d_up_hidden, d_u),
            nn.ReLU(),
        )

        # head
        in_dim = d_t + d_L + d_s + d_u + d_m + 2  # scalars: cfg, n_loras
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
                upscaler_id, up_has, up_steps, denoise, model_id):

        B = tokens.size(0)

        # helper
        def to_col(x, B):
            """ensure tensor is (B,1). accepts (B,) or (B,1)"""
            return x.view(B, 1) if x.dim() == 1 else x

        model_id = model_id.squeeze(-1) if model_id.dim() > 1 else model_id
        model_id = model_id.long()

        v_dom = self.model_domain_embed(model_id)

        # token path
        token_mask = token_mask.to(self.token_embed.weight.dtype)
        t = self.token_embed(tokens)

        # Inject model_id into tokens
        t = t + self.model_token_proj(v_dom).unsqueeze(1).to(t.dtype)

        t = self.token_phi(t)
        t = t * token_mask.unsqueeze(-1)
        v_tokens = t.sum(1) / token_mask.sum(1, keepdim=True).clamp(min=1)

        # LoRA path
        L_emb = self.lora_embed(lora_ids)
        fused = torch.cat([L_emb, lora_w.unsqueeze(-1).to(L_emb.dtype)], -1)
        fused = self.lora_fuse(fused)
        real = (lora_ids != 0).unsqueeze(-1).to(fused.dtype)
        v_loras = (fused * real).sum(1)

        # sampler / steps path
        sampler_id = sampler_id.squeeze(-1)
        steps_bucket = steps_bucket.squeeze(-1)

        samp_emb = self.sampler_embed(sampler_id)  # (B,d_s)
        bucket_emb = self.bucket_embed(steps_bucket)  # (B,d_b)
        steps_log = to_col(steps_log, B).to(samp_emb.dtype)  # (B,1)

        # Inject model_id into sampler path
        v_dom_s = v_dom.to(samp_emb.dtype)
        samp_cat = torch.cat([samp_emb, steps_log, bucket_emb, v_dom_s], dim=-1)
        v_samp = self.samp_fuse(samp_cat)  # (B,d_s)

        # upscaler path
        upscaler_id = upscaler_id.squeeze(-1)
        up_emb = self.upsc_embed(upscaler_id)  # (B,d_u)

        up_has = to_col(up_has,   B).to(up_emb.dtype)
        up_steps = to_col(up_steps, B).to(up_emb.dtype)
        denoise = to_col(denoise,  B).to(up_emb.dtype)

        # Inject model_id into upscaler path
        v_dom_u = v_dom.to(up_emb.dtype)
        up_cat = torch.cat([up_emb, up_has, up_steps, denoise, v_dom_u], dim=-1)
        v_up = self.up_fuse(up_cat)  # (B,d_u)

        # scalar cols
        cfg_col = to_col(cfg, B).to(v_tokens.dtype)
        n_loras_col = to_col(n_loras, B).to(v_tokens.dtype)

        v_dom = v_dom.to(v_tokens.dtype)

        # concat & head
        z = torch.cat(
            [v_tokens, v_loras, v_samp, v_up,
             cfg_col, n_loras_col, v_dom],
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

    vocab_samplers = 20
    vocab_upscalers = 17

    # sd1.5
    # vocab_tokens = 1232
    # vocab_loras = 150

    # sdxl
    vocab_tokens = 1019
    vocab_loras = 1

    max_cfg_scale = 11.0
    max_loras = 7


    vocab_samplers = 20
    vocab_upscalers = 17

    # default hardcoded vocab sizes just in case, though they should come from args/config
    vocab_tokens = 1376
    vocab_loras = 136

    # instantiate & load
    model = PromptQualityModelV3(
        vocab_tokens=vocab_tokens,
        vocab_loras=vocab_loras,
        vocab_samplers=vocab_samplers,
        vocab_upscalers=vocab_upscalers,
        bucket_size=20,
        d_t=256,
        d_L=128,
        head_h1=512,
        dropout_p=0.4,
    )

    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    model.eval()

    return model
