import math
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from .load_dataset import split_dataset
from ..tokenizer import Tokenizer


def pad_sequences(
    sequences,
    maxlen=None,
    dtype="int32",
    padding="pre",
    truncating="pre",
    value=0.0,
):
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(value)}\nYou should set `dtype=object` for variable length "
            "strings."
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x


def parse_loras_and_text(text, tokenizer):
    """
    Extracts LoRAs from text string and returns cleaned text and list of (lora_token_id, weight).
    Assumes LoRAs are in the format "lora name:weight" or "<lora:name:weight>" in the comma-separated string.
    """
    parts = text.split(",")
    clean_parts = []
    loras = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        is_lora = False
        lora_name = ""
        weight = 1.0

        # Check for "lora name:weight" format (standardized by load_dataset)
        if part.startswith("lora "):
            is_lora = True
            try:
                content = part[5:]  # remove "lora "
                if ":" in content:
                    lora_name, weight_str = content.rsplit(":", 1)
                    weight = float(weight_str)
                else:
                    lora_name = content
            except Exception:
                clean_parts.append(part)
                continue
        # Check for <lora:name:weight> format (just in case)
        elif part.startswith("<lora:") and part.endswith(">"):
            is_lora = True
            try:
                content = part[6:-1]
                if ":" in content:
                    lora_name, weight_str = content.rsplit(":", 1)
                    weight = float(weight_str)
                else:
                    lora_name = content
            except Exception:
                clean_parts.append(part)
                continue

        if is_lora:
            token_key = f"lora {lora_name}"
            lora_id = tokenizer.lora_to_token(token_key)

            if lora_id == 0:
                lora_id = tokenizer.lora_to_token(lora_name)

            if lora_id != 0:
                loras.append((lora_id, weight))
        else:
            clean_parts.append(part)

    clean_text = ", ".join(clean_parts)
    return clean_text, loras


def process_sequences_clip_embed(
    _sequences,
    tokenizer,
    clip_tokenizer,
    clip_model,
    embed_cache=None,
    max_length=77,
    device="cpu",
    clip_batch_size=256,
):
    STEP_BUCKET_WIDTH = 5

    labels = []
    lora_ids_list = []
    lora_weights_list = []
    clip_emb_list = []
    tags_list = []

    cfg_scales = []
    n_loras = []
    samplers = []
    steps_raw = []
    steps_log = []
    steps_bucket = []
    upscalers = []
    upscale_steps = []
    denoise_strengths = []
    model_ids = []

    if embed_cache is None:
        embed_cache = {}

    invlid_cfg_cnt = 0

    for _seq in _sequences:
        if _seq["cfg_scale"] > 11.0:
            invlid_cfg_cnt += 1
            continue

        raw_text = _seq["sequence"]

        # Parse LoRAs and clean text
        clean_text, loras = parse_loras_and_text(raw_text, tokenizer)

        tags = [t.strip() for t in clean_text.split(",") if t.strip()]
        tags_list.append(tags)

        # Process LoRAs
        l_ids = [l[0] for l in loras]
        l_weights = [l[1] for l in loras]

        if len(l_ids) == 0:
            pass

        lora_ids_list.append(l_ids)
        lora_weights_list.append(l_weights)
        n_loras.append(len(l_ids))

        cfg_scales.append(_seq["cfg_scale"])
        steps_raw.append(_seq["steps"])
        samplers.append(tokenizer.sampler_to_token(_seq["sampler"]))
        steps_log.append(math.log1p(_seq["steps"]) / math.log1p(60))
        steps_bucket.append(_seq["steps"] // STEP_BUCKET_WIDTH)
        upscale_steps.append(_seq["upscaler_steps"] / 25.0)

        denoise_strengths.append(_seq["denoising_strength"])
        model_ids.append(_seq["model_id"])

        upscalers.append(tokenizer.upscaler_to_token(_seq["upscaler"]))

        if "label" in _seq:
            labels.append(_seq["label"])
        else:
            raise Exception("no Label")
        
    print(f"Skipped {invlid_cfg_cnt} prompts (invalid CFG)")

    # Batch compute missing tag embeddings
    unique_tags = sorted({t for tags in tags_list for t in tags})
    missing_tags = [t for t in unique_tags if t not in embed_cache]

    for start in range(0, len(missing_tags), clip_batch_size):
        batch_tags = missing_tags[start:start + clip_batch_size]
        enc = clip_tokenizer(
            batch_tags,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = clip_model(input_ids=input_ids, attention_mask=attention_mask)
            t = outputs.last_hidden_state  # (N, T, d_t)
            mask = attention_mask.unsqueeze(-1).to(t.dtype)
            pooled_tags = (t * mask).sum(1) / mask.sum(1).clamp(min=1)

        pooled_tags = pooled_tags.detach().cpu().numpy().astype(np.float32)
        for tag, emb in zip(batch_tags, pooled_tags):
            embed_cache[tag] = emb

    d_t = clip_model.config.hidden_size
    for tags in tags_list:
        if len(tags) == 0:
            pooled = np.zeros((d_t,), dtype=np.float32)
        else:
            pooled = np.stack([embed_cache[t] for t in tags], axis=0).mean(0)
        clip_emb_list.append(pooled)

    up_has = [1 if u > 0 else 0 for u in upscalers]

    return (
        np.array(clip_emb_list),
        lora_ids_list,
        lora_weights_list,
        cfg_scales,
        n_loras,
        samplers,
        steps_raw,
        steps_log,
        steps_bucket,
        upscalers,
        upscale_steps,
        denoise_strengths,
        up_has,
        model_ids,
        labels,
        embed_cache,
    )


def preprocess_clip_embed(
    dataset_input,
    dataset_output,
    tokenizer_file,
    clip_model_name="openai/clip-vit-large-patch14",
    device=None,
    cache_path=None,
    clip_batch_size=256,
):

    # Load existing tokenizer for LoRAs/Samplers/Upscalers
    try:
        tokenizer = Tokenizer.load_from_file(tokenizer_file)
        print("Loaded existing tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP tokenizer + text encoder
    print(f"Loading CLIP tokenizer: {clip_model_name}")
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    print(f"Loading CLIP text encoder: {clip_model_name}")
    clip_model = CLIPTextModel.from_pretrained(clip_model_name)
    clip_model.eval()
    clip_model.to(device)

    embed_cache = {}
    cache_file = None
    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            embed_cache = np.load(cache_file, allow_pickle=True).item()
        else:
            cache_file.parent.mkdir(parents=True, exist_ok=True)

    (x_train, y_train, x_val, y_val, x_test, y_test) = split_dataset(
        dataset_input, dataset_output
    )

    x_train = np.concatenate([x_train, x_test], axis=0)
    y_train = np.concatenate([y_train, y_test], axis=0)

    # attach labels
    x_train_with_labels = [{**x, "label": y} for x, y in zip(x_train, y_train)]
    x_val_with_labels = [{**x, "label": y} for x, y in zip(x_val, y_val)]

    print("Processing training sequences...")
    (
        clip_emb_train,
        loras_train,
        weights_train,
        cfg_scales_train,
        n_loras_train,
        samplers_train,
        steps_raw_train,
        steps_log_train,
        steps_bucket_train,
        upscalers_train,
        upscale_steps_train,
        denoise_strengths_train,
        up_has_train,
        model_ids_train,
        y_train,
        embed_cache,
    ) = process_sequences_clip_embed(
        x_train_with_labels,
        tokenizer,
        clip_tokenizer,
        clip_model,
        embed_cache=embed_cache,
        device=device,
        clip_batch_size=clip_batch_size,
    )

    print("Processing validation sequences...")
    (
        clip_emb_val,
        loras_val,
        weights_val,
        cfg_scales_val,
        n_loras_val,
        samplers_val,
        steps_raw_val,
        steps_log_val,
        steps_bucket_val,
        upscalers_val,
        upscale_steps_val,
        denoise_strengths_val,
        up_has_val,
        model_ids_val,
        y_val,
        embed_cache,
    ) = process_sequences_clip_embed(
        x_val_with_labels,
        tokenizer,
        clip_tokenizer,
        clip_model,
        embed_cache=embed_cache,
        device=device,
        clip_batch_size=clip_batch_size,
    )

    if cache_file is not None:
        np.save(cache_file, embed_cache, allow_pickle=True)

    # Determine max loras for padding
    max_loras = 0
    for l in loras_train + loras_val:
        if len(l) > max_loras:
            max_loras = len(l)

    # Ensure at least 1 to avoid empty array issues
    max_loras = max(max_loras, 1)
    print("max loras:", max_loras)

    # Pad LoRAs
    padded_loras_train = pad_sequences(
        loras_train, maxlen=max_loras, padding="post", truncating="post"
    )
    padded_weights_train = pad_sequences(
        weights_train, maxlen=max_loras, padding="post", truncating="post", dtype="float32"
    )

    padded_loras_val = pad_sequences(
        loras_val, maxlen=max_loras, padding="post", truncating="post"
    )
    padded_weights_val = pad_sequences(
        weights_val, maxlen=max_loras, padding="post", truncating="post", dtype="float32"
    )

    # Normalize scalars
    max_cfg_scale = 0.0
    for cfg in cfg_scales_train + cfg_scales_val:
        if cfg > max_cfg_scale:
            max_cfg_scale = cfg

    max_cfg_scale = max(max_cfg_scale, 11.0)

    print("max cfg scale:", max_cfg_scale)

    n_loras_train = [n / max_loras for n in n_loras_train]
    n_loras_val = [n / max_loras for n in n_loras_val]
    cfg_scales_train = [cfg / max_cfg_scale for cfg in cfg_scales_train]
    cfg_scales_val = [cfg / max_cfg_scale for cfg in cfg_scales_val]

    train_samples = [
        clip_emb_train,
        padded_loras_train,
        padded_weights_train,
        cfg_scales_train,
        n_loras_train,
        samplers_train,
        steps_log_train,
        steps_bucket_train,
        upscalers_train,
        up_has_train,
        upscale_steps_train,
        denoise_strengths_train,
        model_ids_train,
        y_train,
    ]
    val_samples = [
        clip_emb_val,
        padded_loras_val,
        padded_weights_val,
        cfg_scales_val,
        n_loras_val,
        samplers_val,
        steps_log_val,
        steps_bucket_val,
        upscalers_val,
        up_has_val,
        upscale_steps_val,
        denoise_strengths_val,
        model_ids_val,
        y_val,
    ]

    vocab_loras = len(tokenizer.lora_index) + 1
    vocab_samplers = len(tokenizer.sampler_index) + 1
    vocab_upscalers = len(tokenizer.upscaler_index) + 1

    return (train_samples, val_samples, vocab_loras, vocab_samplers, vocab_upscalers)

