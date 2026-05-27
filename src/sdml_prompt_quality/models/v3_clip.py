import torch
from torch import nn
from transformers import CLIPTextModel

class PromptQualityModelV3Clip(nn.Module):
    """
    Prompt-quality scorer that extends V3 but replaces the token embedding layer
    with a frozen CLIP text encoder (SDXL TE1 / SD1.5 variant: openai/clip-vit-large-patch14).

    Interface (all tensors are batch-first):
        input_ids      : (B, T)    int64    (CLIP tokens)
        attention_mask : (B, T)    int64    (CLIP mask)
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
        model_id       : (B,)      int64

    Output:
        logits         : (B,)      float32   (apply sigmoid in prod)
    """

    def __init__(
        self,
        vocab_loras: int,
        vocab_samplers: int,
        vocab_upscalers: int,
        vocab_models: int = 2,
        bucket_size: int = 10,
        # dimensions
        d_L: int = 64,
        d_s: int = 32,
        d_u: int = 32,
        d_token_hidden: int = 64,
        d_proj: int = 256,  # Project CLIP embeddings to this dim
        d_lora_hidden: int = 32,
        d_samp_hidden: int = 32,
        d_up_hidden: int = 32,
        d_bucket_emb: int = 8,
        head_h1: int = 128,  # Reduced from 320
        head_h2: int = 64,   # Reduced from 160
        dropout_p: float = 0.5, 
        clip_model_name: str = "openai/clip-vit-large-patch14",
    ):
        super().__init__()

        d_m = 32
        self.d_m = d_m 

        # CLIP Text Encoder
        # use the hidden size from the config, usually 768 for ViT-L/14
        self.clip = CLIPTextModel.from_pretrained(clip_model_name)
        self.clip.requires_grad_(False) # Freeze CLIP
        d_t = self.clip.config.hidden_size

        # embeddings
        self.lora_embed = nn.Embedding(vocab_loras, d_L, padding_idx=0)
        self.sampler_embed = nn.Embedding(vocab_samplers, d_s, padding_idx=0)
        self.bucket_embed = nn.Embedding(bucket_size, d_bucket_emb)
        self.upsc_embed = nn.Embedding(vocab_upscalers, d_u, padding_idx=0)
        self.model_domain_embed = nn.Embedding(vocab_models, d_m)

        # token φ
        # Project model domain embedding to CLIP hidden size
        self.model_token_proj = nn.Linear(d_m, d_t)
        
        # Project CLIP embeddings (768) down to d_proj (256)
        # acts as a bottleneck and regularization
        self.token_phi = nn.Sequential(
            nn.Linear(d_t, d_proj),
            nn.Dropout(0.2), # dropout before ReLU/Projection
            nn.ReLU(),
            # No expansion back to d_t
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
        # input dim is now d_proj instead of d_t
        in_dim = d_proj + d_L + d_s + d_u + d_m + 2  # scalars: cfg, n_loras
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

        # init custom layers (CLIP is already init)
        for emb in (
            self.lora_embed,
            self.sampler_embed,
            self.upsc_embed,
            self.bucket_embed,
            self.model_domain_embed,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self,
                input_ids, attention_mask,
                lora_ids, lora_w,
                cfg, n_loras,
                sampler_id, steps_log, steps_bucket,
                upscaler_id, up_has, up_steps, denoise, model_id):

        B = input_ids.size(0)

        # helper
        def to_col(x, B):
            """ensure tensor is (B,1). accepts (B,) or (B,1)"""
            return x.view(B, 1) if x.dim() == 1 else x

        model_id = model_id.squeeze(-1) if model_id.dim() > 1 else model_id
        model_id = model_id.long()

        v_dom = self.model_domain_embed(model_id)

        # token path (CLIP)
        # use the last hidden state to get per-token embeddings
        with torch.no_grad():
            outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask)
            t = outputs.last_hidden_state # (B, T, d_t)

        # Inject model_id into tokens
        # Project v_dom to d_t and add
        t = t + self.model_token_proj(v_dom).unsqueeze(1).to(t.dtype)

        # Project down to d_proj
        t = self.token_phi(t) # (B, T, d_proj)
        
        # Masking and averaging
        # attention_mask is 1 for tokens, 0 for padding
        mask_expanded = attention_mask.unsqueeze(-1).to(t.dtype)
        t = t * mask_expanded
        v_tokens = t.sum(1) / mask_expanded.sum(1).clamp(min=1)

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
