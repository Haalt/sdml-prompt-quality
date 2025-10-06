import json
import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re


def parseTag(tag):
    try:
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
    query = """
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


def load_sqlite_model_scores(database_path, map={}, include_null=False):
    """Load dataset with model scores from database"""
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

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
        query = """
        SELECT
            m.metadata_content,
            i.generation_info,
            AVG(i.model_score)          AS mean_score,
            COUNT(*)                    AS count
        FROM metadata AS m
        JOIN images   AS i ON i.metadata_id = m.metadata_id
        WHERE i.model_score IS NOT NULL
        GROUP BY m.metadata_id;
        """
        # GROUP BY m.metadata_id, i.generation_info;

    cursor.execute(query)
    records = cursor.fetchall()

    cfg_scale_pattern = r"CFG scale:\s*([0-9]*\.?[0-9]+)"
    sampler_pattern = r"Sampler: (.*?(?=,))"
    steps_pattern = r"Steps: ([0-9]+)(?=,)"
    upscaler_pattern = r"Hires upscaler: (.*?(?=,))(?=,)"
    upscaler_steps_pattern = r"Hires steps: ([0-9]+)(?=,)"
    denoising_strength_pattern = r"Denoising strength: ([0|1]\.[0-9]+)(?=,)"

    # group by prompt and CFG scale
    grouped_scores = {}

    for record in records:
        # metadata_id, metadata_content, generation_info, status, model_score = record
        metadata_content, generation_info, mean_score, count = record

        if generation_info is None or generation_info == '':
            continue

        # extract CFG scale from generation_info
        cfg_match = re.search(cfg_scale_pattern, generation_info)
        cfg_scale = float(cfg_match.group(
            1)) if cfg_match else 7.0  # default CFG scale

        try:

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
            print(generation_info)
            continue

        tags = "".join(metadata_content.split("\n")[:-1]).split(",")
        tags = [parseTag(tag.strip()) for tag in tags]
        key = ",".join(tags)

        # Create a unique key that includes CFG scale
        key_with_cfg = f"{key}||cfg:{cfg_scale}"

        # Determine score
        if mean_score is not None:
            score = mean_score
        # elif include_null:
        #     # Fallback to original label for null scores
        #     score = 1.0 if status == 'saved' else 0.0
        else:
            # Skip if no model score and not including null
            continue

        # Collect scores for this prompt+cfg combination
        if key_with_cfg not in grouped_scores:
            map[key_with_cfg] = {
                "mean_score": score,
                "sequence": key,
                "cfg_scale": cfg_scale,
                "sampler": sampler,
                "steps": steps,
                "upscaler": upscaler,
                "upscaler_steps": upscaler_steps,
                "denoising_strength": denoising_strength,
            }
        # grouped_scores[key_with_cfg]["scores"].append(score)

    # Calculate mean scores for each group
    # for key_with_cfg, data in grouped_scores.items():
    #     scores = data["scores"]
    #     mean_score = sum(scores) / len(scores) if scores else 0.0

    #     map[key_with_cfg] = {
    #         "mean_score": mean_score,
    #         "count": len(scores),
    #         "sequence": data["sequence"],
    #         "cfg_scale": data["cfg_scale"]
    #     }

    conn.close()
    return map


def load_dataset_model_scores(include_null=False):
    """Load dataset with model scores instead of binary labels"""
    map = {}

    try:
        load_sqlite_model_scores(
            "./dataset/sd.db", map, include_null=include_null)
    except sqlite3.OperationalError:
        load_sqlite_model_scores(
            "../dataset/sd.db", map, include_null=include_null)

    dataset_input = []
    dataset_output = []

    for key, value in map.items():
        if len(value["sequence"].split(",")) <= 82:
            # Create input dict with all features
            dataset_input.append(
                {
                    "sequence": value["sequence"],
                    "cfg_scale": value["cfg_scale"],
                    "sampler": value["sampler"],
                    "steps": value["steps"],
                    "upscaler": value["upscaler"],
                    "upscaler_steps": value["upscaler_steps"],
                    "denoising_strength": value["denoising_strength"],
                }
            )
            dataset_output.append(value["mean_score"])

    print(len(dataset_input), len(dataset_output))

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

    # threshold = 2
    y_series = pd.Series(dataset_output)

    # class_counts = y_series.value_counts()

    # rare_classes = class_counts[class_counts < threshold].index

    y_binned = pd.qcut(y_series, q=10, labels=False, duplicates="drop")

    x_train, x_val, y_train, y_val = train_test_split(
        dataset_input, y_series, test_size=0.2, random_state=42, stratify=y_binned
    )

    x_train, x_val, y_train, y_val = (
        np.array(x_train),
        np.array(x_val),
        np.array(y_train),
        np.array(y_val),
    )

    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

    x, y = int(len(x_val) / 2), int(len(y_val) / 2)
    return x_train, y_train, x_val[x:], y_val[y:], x_val[:x], y_val[:y]
    # return x_train, y_train, x_val, y_val
