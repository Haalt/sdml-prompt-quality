import json
import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re


def parseTag(tag, is_xl):
    try:
        if is_xl:
            return f"lora {tag.split(':')[1]}:{tag.split(':')[2][:-1]}" if "<lora" in tag else tag
        else:
            return f"lora {tag.split(':')[1]}:{tag.split(':')[2][:-1]}" if "<" in tag else tag.replace("(", "").replace(")", "")
    except Exception as e:
        return tag.split(":")[1] if "<" in tag else tag


def load_sqlite(database_path, map={}):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    query = """
    SELECT
        m.metadata_id,
        m.metadata_content,
        SUM(CASE WHEN i.status = 'saved' THEN 1 ELSE 0 END) as saved_count,
        SUM(CASE WHEN i.status = 'deleted' OR i.status = 'binned' THEN 1 ELSE 0 END) as deleted_count
    FROM metadata m
    LEFT JOIN images i ON m.metadata_id = i.metadata_id
    GROUP BY m.metadata_id
    """

    cursor.execute(query)
    records = cursor.fetchall()

    for record in records:
        metadata_id, metadata_content, saved_count, deleted_count = record

        tags = "".join(metadata_content.split("\n")[:-1]).split(",")
        tags = [parseTag(tag.strip()) for tag in tags]
        key = ",".join(tags)
        try:
            map[key]["saved"] += saved_count
            map[key]["deleted"] += deleted_count
        except KeyError:
            map[key] = {"saved": saved_count, "deleted": deleted_count}

    conn.close()
    return map


def load_sqlite_logits(database_path, map={}):
    """Load dataset with CFG scale information from generation_info"""
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # query = """
    # SELECT
    #     m.metadata_id,
    #     m.metadata_content,
    #     i.generation_info,
    #     SUM(CASE WHEN i.status = 'saved' THEN 1 ELSE 0 END) as saved_count,
    #     SUM(CASE WHEN i.status = 'deleted' OR i.status = 'binned' THEN 1 ELSE 0 END) as deleted_count
    # FROM metadata m
    # LEFT JOIN images i ON m.metadata_id = i.metadata_id
    # WHERE i.generation_info IS NOT NULL
    # GROUP BY m.metadata_id, i.generation_info
    # """


    query = f"""
    SELECT
        m.metadata_id,
        m.metadata_content,
        i.generation_info,
        SUM(CASE WHEN i.status = 'saved' THEN 1 ELSE 0 END) as saved_count,
        SUM(CASE WHEN i.status = 'deleted' OR i.status = 'binned' THEN 1 ELSE 0 END) as deleted_count
    FROM metadata m
    LEFT JOIN images i ON m.metadata_id = i.metadata_id
    GROUP BY m.metadata_id
    """

    cursor.execute(query)
    records = cursor.fetchall()

    cfg_scale_pattern = r"CFG scale:\s*([0-9]*\.?[0-9]+)"

    for record in records:
        metadata_id, metadata_content, generation_info, saved_count, deleted_count = (
            record
        )

        if generation_info is None:
            continue

        # Extract CFG scale from generation_info
        cfg_match = re.search(cfg_scale_pattern, generation_info)
        cfg_scale = float(cfg_match.group(
            1)) if cfg_match else 7.0  # default CFG scale

        tags = "".join(metadata_content.split("\n")[:-1]).split(",")
        tags = [parseTag(tag.strip()) for tag in tags]
        key = ",".join(tags)

        # create a unique key that includes CFG scale to handle different CFG values for same prompt
        key_with_cfg = f"{key}||cfg:{cfg_scale}"

        try:
            map[key_with_cfg]["saved"] += saved_count
            map[key_with_cfg]["deleted"] += deleted_count
        except KeyError:
            map[key_with_cfg] = {
                "saved": saved_count,
                "deleted": deleted_count,
                "sequence": key,
                "cfg_scale": cfg_scale,
            }

    conn.close()
    return map


def load_dataset():
    map = {}

    try:
        load_sqlite("./dataset/sd.db", map)
    except sqlite3.OperationalError:
        load_sqlite("../dataset/sd.db", map)

    dataset_input = []
    dataset_output = []

    for key, value in map.items():
        if len(key.split(",")) <= 82:
            dataset_input.append(key)
            dataset_output.append(
                value["saved"] / (value["saved"] + value["deleted"] or 1)
            )

    return (dataset_input, dataset_output)


def load_dataset_logits():
    """Load dataset with CFG scale information"""
    map = {}

    try:
        load_sqlite_logits("./dataset/sd.db", map)
    except sqlite3.OperationalError:
        load_sqlite_logits("../dataset/sd.db", map)

    dataset_input = []
    dataset_output = []

    for key, value in map.items():
        # if value["saved"] + value["deleted"] % 8 != 0:
        #     continue
        if len(value["sequence"].split(",")) <= 82:
            # Create input dict with sequence and cfg_scale
            dataset_input.append(
                {"sequence": value["sequence"],
                    "cfg_scale": value["cfg_scale"]}
            )
            dataset_output.append(
                value["saved"] / (value["saved"] + value["deleted"] or 1)
            )

    return (dataset_input, dataset_output)


def load_sqlite_model_scores(database_path, map={}, include_null=False, is_xl=False):
    """Load dataset with model scores from database"""
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    if is_xl:
        MODEL_NAME = 'sdxl'
    else:
        MODEL_NAME = 'sd1.5'


    if include_null:
        # Include all images, using original labels for null scores
        # query = """
        # SELECT
        #     m.metadata_id,
        #     m.metadata_content,
        #     i.generation_info,
        #     i.status,
        #     i.model_score
        # FROM metadata m
        # LEFT JOIN images i ON m.metadata_id = i.metadata_id
        # """
        pass
    else:
        # Only include images with model scores
        query = f"""
        SELECT
            m.metadata_content,
            i.generation_info,
            i.model_score
        FROM metadata AS m
        JOIN images   AS i ON i.metadata_id = m.metadata_id
        WHERE i.model_score IS NOT NULL AND i.generation_info LIKE '%Model: {MODEL_NAME}%'
        """


    cursor.execute(query)
    records = cursor.fetchall()

    cfg_scale_pattern = r"CFG scale:\s*([0-9]*\.?[0-9]+)"
    sampler_pattern = r"Sampler: (.*?(?=,))"
    steps_pattern = r"Steps: ([0-9]+)(?=,)"
    upscaler_pattern = r"Hires upscaler: (.*?(?=,))(?=,)"
    upscaler_steps_pattern = r"Hires steps: ([0-9]+)(?=,)"
    denoising_strength_pattern = r"Denoising strength: ([0|1]\.[0-9]+)(?=,)"
    loras_pattern = r'Lora hashes: "([^"]*)"'

    # Group by prompt + CFG + generation settings to build empirical quantiles.
    grouped_scores = {}

    for record in records:
        metadata_content, generation_info, model_score = record

        if generation_info is None or generation_info == '':
            continue

        # extract CFG scale from generation_info
        cfg_match = re.search(cfg_scale_pattern, generation_info)
        cfg_scale = float(cfg_match.group(
            1)) if cfg_match else 7.0  # default CFG scale

        try:
            loras_match = re.search(loras_pattern, generation_info)
            loras = loras_match.group(1) if loras_match else ""
            loras = ["lora " + lora.split(":")[0].strip() for lora in loras.split(",") if len(loras) > 0]

            sampler_match = re.search(sampler_pattern, generation_info)
            sampler = sampler_match.group(1)
            sampler = sampler.replace(" Karras", "")  # SD schedulers update

            steps_match = re.search(steps_pattern, generation_info)
            steps = int(steps_match.group(1))

            upscaler_match = re.search(upscaler_pattern, generation_info)
            upscaler = upscaler_match.group(1) if upscaler_match else "None"
            upscaler_steps_match = re.search(
                upscaler_steps_pattern, generation_info)
            upscaler_steps = int(
                upscaler_steps_match.group(1) if upscaler_steps_match else 0
            )
            denoising_strength_match = re.search(
                denoising_strength_pattern, generation_info
            )
            denoising_strength = (
                float(denoising_strength_match.group(1))
                if denoising_strength_match
                else 0.0
            )
        except:
            # print(generation_info)
            continue

        tags = "".join(metadata_content.split("\n")[:-1]).split(",")
        tags = [parseTag(tag.strip(), is_xl) for tag in tags]
        key = ",".join(tags)

        # Create a unique key that includes CFG scale
        key_with_cfg = f"{key}||cfg:{cfg_scale}"

        if model_score is None:
            continue

        score = float(model_score)

        # Collect scores for this prompt+cfg combination
        if key_with_cfg not in grouped_scores:
            grouped_scores[key_with_cfg] = {
                "scores": [score],
                "loras": loras,
                "sequence": key,
                "cfg_scale": cfg_scale,
                "sampler": sampler,
                "steps": steps,
                "upscaler": upscaler,
                "upscaler_steps": upscaler_steps,
                "denoising_strength": denoising_strength,
            }
        else:
            grouped_scores[key_with_cfg]["scores"].append(score)

    for key_with_cfg, data in grouped_scores.items():
        scores = np.asarray(data["scores"], dtype=np.float32)
        if scores.size == 0:
            continue
        map[key_with_cfg] = {
            "mean_score": float(np.mean(scores)),
            "q50_score": float(np.quantile(scores, 0.5)),
            "q90_score": float(np.quantile(scores, 0.9)),
            "count": int(scores.size),
            "raw_scores": data["scores"],
            "loras": data["loras"],
            "sequence": data["sequence"],
            "cfg_scale": data["cfg_scale"],
            "sampler": data["sampler"],
            "steps": data["steps"],
            "upscaler": data["upscaler"],
            "upscaler_steps": data["upscaler_steps"],
            "denoising_strength": data["denoising_strength"],
        }

    conn.close()
    return map


def load_dataset_model_scores(
    include_null=False,
    is_xl=False,
    combine=False,
    normalize=True,
    target_mode="mean",
):
    """Load dataset with model scores instead of binary labels"""
    
    def _get_data(is_xl_flag):
        map_data = {}
        try:
            load_sqlite_model_scores(
                "./dataset/sd.db", map_data, include_null=include_null, is_xl=is_xl_flag)
        except sqlite3.OperationalError:
            load_sqlite_model_scores(
                "../dataset/sd.db", map_data, include_null=include_null, is_xl=is_xl_flag)
        
        inputs = []
        scores = []
        
        for key, value in map_data.items():
            if len(value["sequence"].split(",")) <= 82:
                inputs.append({
                    "sequence": value["sequence"],
                    "loras": value["loras"],
                    "cfg_scale": value["cfg_scale"],
                    "sampler": value["sampler"],
                    "steps": value["steps"],
                    "upscaler": value["upscaler"],
                    "upscaler_steps": value["upscaler_steps"],
                    "denoising_strength": value["denoising_strength"],
                    "model_id": 1 if is_xl_flag else 0
                })
                if target_mode == "quantiles":
                    scores.append([value["q50_score"], value["q90_score"]])
                else:
                    scores.append(value["mean_score"])
        
        return inputs, scores

    if combine:
        # Load both datasets
        print("Loading SD1.5 dataset...")
        inputs_sd15, scores_sd15 = _get_data(False)
        print(f"Loaded {len(scores_sd15)} SD1.5 samples")
        
        print("Loading SDXL dataset...")
        inputs_xl, scores_xl = _get_data(True)
        print(f"Loaded {len(scores_xl)} SDXL samples")
        
        if normalize:
            print("Normalizing scores separately (Z-score)...")
            # Normalize SD1.5
            s15 = np.array(scores_sd15, dtype=np.float32)
            if len(s15) > 0:
                if target_mode == "quantiles":
                    mean_15 = np.mean(s15, axis=0)
                    std_15 = np.std(s15, axis=0)
                    print(
                        f"SD1.5: mean_q50={mean_15[0]:.4f}, std_q50={std_15[0]:.4f}, "
                        f"mean_q90={mean_15[1]:.4f}, std_q90={std_15[1]:.4f}"
                    )
                else:
                    mean_15, std_15 = np.mean(s15), np.std(s15)
                    print(f"SD1.5: mean={mean_15:.4f}, std={std_15:.4f}")
                s15_norm = (s15 - mean_15) / (std_15 + 1e-8)
                scores_sd15 = s15_norm.tolist()

            # Normalize SDXL
            sxl = np.array(scores_xl, dtype=np.float32)
            if len(sxl) > 0:
                if target_mode == "quantiles":
                    mean_xl = np.mean(sxl, axis=0)
                    std_xl = np.std(sxl, axis=0)
                    print(
                        f"SDXL: mean_q50={mean_xl[0]:.4f}, std_q50={std_xl[0]:.4f}, "
                        f"mean_q90={mean_xl[1]:.4f}, std_q90={std_xl[1]:.4f}"
                    )
                else:
                    mean_xl, std_xl = np.mean(sxl), np.std(sxl)
                    print(f"SDXL: mean={mean_xl:.4f}, std={std_xl:.4f}")
                sxl_norm = (sxl - mean_xl) / (std_xl + 1e-8)
                scores_xl = sxl_norm.tolist()
        
        dataset_input = inputs_sd15 + inputs_xl
        dataset_output = scores_sd15 + scores_xl
        
    else:
        # Load single dataset
        dataset_input, dataset_output = _get_data(is_xl)
        
        if normalize:
            print(f"Normalizing scores (is_xl={is_xl})...")
            s = np.array(dataset_output, dtype=np.float32)
            if len(s) > 0:
                if target_mode == "quantiles":
                    mean_val = np.mean(s, axis=0)
                    std_val = np.std(s, axis=0)
                    print(
                        f"Stats: mean_q50={mean_val[0]:.4f}, std_q50={std_val[0]:.4f}, "
                        f"mean_q90={mean_val[1]:.4f}, std_q90={std_val[1]:.4f}"
                    )
                else:
                    mean_val, std_val = np.mean(s), np.std(s)
                    print(f"Stats: mean={mean_val:.4f}, std={std_val:.4f}")
                s_norm = (s - mean_val) / (std_val + 1e-8)
                dataset_output = s_norm.tolist()

    print(f"Total samples: {len(dataset_input)}")
    return (dataset_input, dataset_output)


def split_dataset(dataset_input, dataset_output):
    def get_min(arr):
        m = 1.0
        index = 0
        for i in range(len(arr.index)):
            if arr.index[i] <= m:
                m = arr.index[i]
                index = i
        return index

    y_np = np.asarray(dataset_output)
    if y_np.ndim > 1:
        y_strat = y_np[:, 0]
    else:
        y_strat = y_np
    y_series = pd.Series(y_strat)

    # class_counts = y_series.value_counts()

    # rare_classes = class_counts[class_counts < threshold].index

    y_binned = pd.qcut(y_series, q=10, labels=False, duplicates="drop")

    x_train, x_val, y_train_idx, y_val_idx = train_test_split(
        np.asarray(dataset_input),
        np.arange(len(dataset_input)),
        test_size=0.2,
        random_state=42,
        stratify=y_binned,
    )

    y_train = y_np[y_train_idx]
    y_val = y_np[y_val_idx]

    x_train, x_val, y_train, y_val = (
        np.array(x_train, dtype=object),
        np.array(x_val, dtype=object),
        np.array(y_train),
        np.array(y_val),
    )

    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

    x, y = int(len(x_val) / 2), int(len(y_val) / 2)
    return x_train, y_train, x_val[x:], y_val[y:], x_val[:x], y_val[:y]
    # return x_train, y_train, x_val, y_val
