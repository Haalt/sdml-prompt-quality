import json
import numpy as np
import math
from pathlib import Path


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

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

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


def process_sequences(_sequences, tokenizer):
    loras = []
    sequences = []
    sequences_masks = []
    weights = []
    cfg_scales = []
    n_loras = []
    samplers = []
    steps_raw = []
    steps_log = []
    steps_bucket = []
    upscalers = []
    upscale_steps = []
    denoise_strengths = []

    STEP_BUCKET_WIDTH = 5

    labels = []

    for _seq in _sequences:
        seq = _seq["sequence"]
        w, s, l, s_m = ([], [], [], [])
        n_l = 0
        for token, weight in seq:
            if tokenizer.token_to_text(token).startswith("lora"):
                # s.append(token)
                w.append(weight)
                # l.append(token)
                l.append(tokenizer.lora_to_token(
                    tokenizer.token_to_text(token)))
                n_l += 1
            else:
                s.append(token)
                s_m.append(1)

        loras.append(l)
        weights.append(w)
        sequences.append(s)
        sequences_masks.append(s_m)
        n_loras.append(n_l)
        cfg_scales.append(_seq["cfg_scale"])  # TODO: normalize
        steps_raw.append(_seq["steps"])
        samplers.append(tokenizer.sampler_to_token(_seq["sampler"]))
        steps_log.append(math.log1p(_seq["steps"]) / math.log1p(60))
        steps_bucket.append(_seq["steps"] // STEP_BUCKET_WIDTH)
        upscale_steps.append(_seq["upscaler_steps"] / 25.0)

        denoise_strengths.append(_seq["denoising_strength"])

        upscalers.append(tokenizer.upscaler_to_token(_seq["upscaler"]))
        # keep label aligned with kept sample
        if "label" in _seq:
            labels.append(_seq["label"])
        else:
            raise Exception("no Label")

    up_has = [1 if u > 0 else 0 for u in upscalers]

    return (
        sequences,
        sequences_masks,
        loras,
        weights,
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
        labels
    )


def preprocess(dataset_input, dataset_output, tokenizer_file):
    try:
        bad_file = str(Path(__file__).parent / "bad.json")
        print(bad_file)
        with open(bad_file, "r") as f:
            bad_arr = json.load(f)
            print("Loaded bad array")
    except:
        print("No bad array found")

    (x_train, y_train, x_val, y_val, x_test, y_test) = split_dataset(
        dataset_input, dataset_output
    )

    x_train = np.concatenate([x_train, x_test], axis=0)
    y_train = np.concatenate([y_train, y_test], axis=0)

    x_train_tmp = x_train
    y_train_tmp = y_train
    x_val_tmp = x_val
    y_val_tmp = y_val

    x_train, y_train, x_val, y_val = [], [], [], []

    for x, y in zip(x_train_tmp, y_train_tmp):
        if x["sequence"] not in bad_arr:
            x_train.append(x)
            y_train.append(y)

    for x, y in zip(x_val_tmp, y_val_tmp):
        if x["sequence"] not in bad_arr:
            x_val.append(x)
            y_val.append(y)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    try:
        tokenizer = Tokenizer.load_from_file(tokenizer_file)
        # tokenizer.fit_on_texts(np.concatenate((x_train, x_val), axis=None))

        print("Loaded tokenizer")
    except Exception as e:
        print(e)
        print("Couldn't load tokenizer, fitting on text...")
        tokenizer = Tokenizer(split=",")
        tokenizer.fit_on_texts(np.concatenate(
            (x_train, x_val, x_test), axis=None))
        tokenizer.fit_on_loras(np.concatenate(
            (x_train, x_val, x_test), axis=None))

        print("Saving tokenizer")
        tokenizer.save(tokenizer_file)

    vocab_size = tokenizer.length
    lora_vocab_size = len(tokenizer.lora_index)
    print(vocab_size)

    # attach labels so downstream drops preserve alignment
    x_train_with_labels = [{**x, "label": y} for x, y in zip(x_train, y_train)]
    x_val_with_labels = [{**x, "label": y} for x, y in zip(x_val, y_val)]

    sequences = tokenizer.texts_to_sequences(x_train_with_labels)
    sequences_val = tokenizer.texts_to_sequences(x_val_with_labels)
    # sequences_test = tokenizer.texts_to_sequences(x_test)

    (
        sequences,
        sequences_masks,
        loras,
        weights,
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
        y_train,
    ) = process_sequences(sequences, tokenizer)

    (
        sequences_val,
        sequences_masks_val,
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
        y_val,
    ) = process_sequences(sequences_val, tokenizer)

    # (
    #     sequences_test,
    #     sequences_masks_test,
    #     loras_test,
    #     weights_test,
    #     cfg_scales_test,
    #     n_loras_test,
    #     samplers_test,
    #     steps_raw_test,
    #     steps_log_test,
    #     steps_bucket_test,
    #     upscalers_test,
    #     upscale_steps_test,
    #     denoise_strengths_test,
    #     up_has_test,
    # ) = process_sequences(sequences_test, tokenizer)

    max_loras = 0
    nb_loras = set()
    for l in loras + loras_val:  # + loras_test:
        if len(l) > max_loras:
            max_loras = len(l)
        for _l in l:
            nb_loras.add(_l)

    nb_loras = len(nb_loras)
    print("max loras:", max_loras)
    print("nb loras:", nb_loras)

    sampler_vocab_size = len(tokenizer.sampler_index)
    upscaler_vocab_size = len(tokenizer.upscaler_index)

    print("sampler vocab size:", sampler_vocab_size)
    print("upscaler vocab size:", upscaler_vocab_size)

    weight_set = set()
    for seq in weights + weights_val:  # + weights_test:
        for weight in seq:
            weight_set.add(int(weight * 100.0))

    max_length = 0
    for s in sequences + sequences_val:  # + sequences_test:
        if len(s) > max_length:
            max_length = len(s)

    max_cfg_scale = 0.0
    for cfg in cfg_scales + cfg_scales_val:  # + cfg_scales_test:
        if cfg > max_cfg_scale:
            max_cfg_scale = cfg

    print(f"max cfg scale: {max_cfg_scale}")

    print("max length:", max_length)

    padded_sequences_token_train = pad_sequences(
        sequences, maxlen=max_length, padding="post", truncating="post"
    )
    padded_sequences_masks_train = pad_sequences(
        sequences_masks, maxlen=max_length, padding="post", truncating="post"
    )
    padded_sequences_weight_train = pad_sequences(
        weights, maxlen=max_loras, padding="post", truncating="post", dtype="float32"
    )
    padded_sequences_loras_train = pad_sequences(
        loras, maxlen=max_loras, padding="post", truncating="post"
    )

    padded_sequences_token_val = pad_sequences(
        sequences_val, maxlen=max_length, padding="post", truncating="post"
    )
    padded_sequences_masks_val = pad_sequences(
        sequences_masks_val, maxlen=max_length, padding="post", truncating="post"
    )
    padded_sequences_weight_val = pad_sequences(
        weights_val,
        maxlen=max_loras,
        padding="post",
        truncating="post",
        dtype="float32",
    )
    padded_sequences_loras_val = pad_sequences(
        loras_val, maxlen=max_loras, padding="post", truncating="post"
    )

    # padded_sequences_token_test = pad_sequences(
    #     sequences_test, maxlen=max_length, padding="post", truncating="post"
    # )
    # padded_sequences_weight_test = pad_sequences(
    #     weights_test,
    #     maxlen=max_loras,
    #     padding="post",
    #     truncating="post",
    #     dtype="float32",
    # )
    # padded_sequences_loras_test = pad_sequences(
    #     loras_test, maxlen=max_loras, padding="post", truncating="post"
    # )

    n_loras = [n / max_loras for n in n_loras]
    n_loras_val = [n / max_loras for n in n_loras_val]
    cfg_scales = [cfg / max_cfg_scale for cfg in cfg_scales]
    cfg_scales_val = [cfg / max_cfg_scale for cfg in cfg_scales_val]

    train_samples = [
        padded_sequences_token_train,
        padded_sequences_masks_train,
        padded_sequences_loras_train,
        padded_sequences_weight_train,
        cfg_scales,
        n_loras,
        samplers,
        steps_log,
        steps_bucket,
        upscalers,
        up_has,
        upscale_steps,
        denoise_strengths,
        y_train,
    ]
    val_samples = [
        padded_sequences_token_val,
        padded_sequences_masks_val,
        padded_sequences_loras_val,
        padded_sequences_weight_val,
        cfg_scales_val,
        n_loras_val,
        samplers_val,
        steps_log_val,
        steps_bucket_val,
        upscalers_val,
        up_has_val,
        upscale_steps_val,
        denoise_strengths_val,
        y_val,
    ]

    print("max token index:", padded_sequences_token_train.max())
    print("expected vocab size:", vocab_size)
    print("max lora index: ", padded_sequences_loras_train.max())
    print("expected lora vocab size:", lora_vocab_size)

    print(
        max(steps_bucket),      # should be < bucket_size
        max(samplers),          # should be < vocab_samplers
        max(upscalers),         # should be < vocab_upscalers
    )

    return (train_samples, val_samples, vocab_size, lora_vocab_size + 1, sampler_vocab_size, upscaler_vocab_size)
