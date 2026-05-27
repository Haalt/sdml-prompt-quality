import json
import numpy as np
import math
from pathlib import Path
from transformers import CLIPTokenizer

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
                content = part[5:] # remove "lora "
                if ":" in content:
                    lora_name, weight_str = content.rsplit(":", 1)
                    weight = float(weight_str)
                else:
                    lora_name = content
            except:
                # Fallback if parsing fails
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
            except:
                clean_parts.append(part)
                continue
        
        if is_lora:
            # Map lora name to ID using existing tokenizer
            # The existing tokenizer stores loras as "name" in lora_index
            # But the input text might have "lora name:weight"
            # load_dataset.py produces "lora name:weight"
            
            # We need to match how Tokenizer.py stored them.
            # Tokenizer.py: _get_token splits by ":" if starts with lora.
            # fit_on_loras: if _token.startswith("lora"): token = _get_token(_token) -> lora_index[token]
            
            # So if the string is "lora name:weight", the token key is "lora name"
            
            token_key = f"lora {lora_name}"
            lora_id = tokenizer.lora_to_token(token_key)
            
            # If not found, try without "lora " prefix just in case
            if lora_id == 0:
                 lora_id = tokenizer.lora_to_token(lora_name)
            
            if lora_id != 0:
                loras.append((lora_id, weight))
        else:
            clean_parts.append(part)
            
    clean_text = ", ".join(clean_parts)
    return clean_text, loras


def process_sequences_clip(_sequences, tokenizer, clip_tokenizer, max_length=77):
    STEP_BUCKET_WIDTH = 5

    labels = []
    lora_ids_list = []
    lora_weights_list = []
    
    input_ids_list = []
    attention_mask_list = []
    
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

    for _seq in _sequences:
        raw_text = _seq["sequence"]
        
        # Parse LoRAs and clean text
        clean_text, loras = parse_loras_and_text(raw_text, tokenizer)
        
        # Tokenize with CLIP
        # truncation=True, padding="max_length", max_length=max_length
        # We will handle padding in batch or here. Let's do it here to return numpy arrays.
        enc = clip_tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np"
        )
        
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        
        # Process LoRAs
        l_ids = [l[0] for l in loras]
        l_weights = [l[1] for l in loras]
        
        if len(l_ids) == 0:
            # Add a dummy lora if none present (will be padded/ignored)
            # Or just leave empty
            pass

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        
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

    up_has = [1 if u > 0 else 0 for u in upscalers]

    return (
        np.array(input_ids_list),
        np.array(attention_mask_list),
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
        labels
    )


def preprocess_clip(dataset_input, dataset_output, tokenizer_file, clip_model_name="openai/clip-vit-large-patch14"):
    
    # Load existing tokenizer for LoRAs/Samplers/Upscalers
    try:
        tokenizer = Tokenizer.load_from_file(tokenizer_file)
        print("Loaded existing tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

    # Load CLIP tokenizer
    print(f"Loading CLIP tokenizer: {clip_model_name}")
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    (x_train, y_train, x_val, y_val, x_test, y_test) = split_dataset(
        dataset_input, dataset_output
    )

    x_train = np.concatenate([x_train, x_test], axis=0)
    y_train = np.concatenate([y_train, y_test], axis=0)

    # Filter bad samples if needed (omitted for brevity, can add back if critical)
    
    # attach labels
    x_train_with_labels = [{**x, "label": y} for x, y in zip(x_train, y_train)]
    x_val_with_labels = [{**x, "label": y} for x, y in zip(x_val, y_val)]

    print("Processing training sequences...")
    (
        input_ids_train,
        attention_mask_train,
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
    ) = process_sequences_clip(x_train_with_labels, tokenizer, clip_tokenizer)

    print("Processing validation sequences...")
    (
        input_ids_val,
        attention_mask_val,
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
    ) = process_sequences_clip(x_val_with_labels, tokenizer, clip_tokenizer)

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
            
    # Avoid division by zero
    # max_cfg_scale = max(max_cfg_scale, 1.0)
    max_cfg_scale = max(max_cfg_scale, 11.0)
    
    n_loras_train = [n / max_loras for n in n_loras_train]
    n_loras_val = [n / max_loras for n in n_loras_val]
    cfg_scales_train = [cfg / max_cfg_scale for cfg in cfg_scales_train]
    cfg_scales_val = [cfg / max_cfg_scale for cfg in cfg_scales_val]

    train_samples = [
        input_ids_train,
        attention_mask_train,
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
        input_ids_val,
        attention_mask_val,
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

