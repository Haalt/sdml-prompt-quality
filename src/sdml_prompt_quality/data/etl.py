import sqlite3
import re
import json
import os
import argparse
from typing import Dict, Optional

config_path = os.path.join(os.path.dirname(__file__), 'config.json')

with open(config_path, "r") as f:
    config = json.load(f)


try:
    BANNED = config["banned"]
except KeyError:
    BANNED = []

# prompt regex configurable via config.json, with sensible default
try:
    PROMPT_REGEX = config["prompt_regex"]
except KeyError:
    # Default matches everything so no filtering occurs if not configured
    PROMPT_REGEX = r".*"

# global variable to hold model predictions for insert-time scoring
MODEL_PREDICTIONS = {}


def load_model_predictions(predictions_file: str) -> Dict[str, float]:
    """Load model predictions from JSON file"""
    global MODEL_PREDICTIONS
    if not predictions_file or not os.path.exists(predictions_file):
        print(f"⚠️  Model predictions file not found: {predictions_file}")
        return {}

    print(f"📂 Loading model predictions from {predictions_file}")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # convert to filename -> score mapping
    MODEL_PREDICTIONS = {
        pred['filename']: pred['softmax_score']
        for pred in predictions
    }

    print(f"✅ Loaded {len(MODEL_PREDICTIONS)} model predictions")
    return MODEL_PREDICTIONS


def get_model_score(image_path: str) -> Optional[float]:
    """Get model score for an image based on its path"""
    if not MODEL_PREDICTIONS:
        return None

    if not image_path:
        return None

    # extract filename from path
    filename = os.path.basename(image_path)

    if filename in MODEL_PREDICTIONS:
        return MODEL_PREDICTIONS[filename]

    # fallback: try to find any filename that matches the end of the path
    for pred_filename, score in MODEL_PREDICTIONS.items():
        if image_path.endswith(pred_filename):
            return score

    return None


def clean_db(db_path, include_model_scores=False):
    output_conn = sqlite3.connect(db_path)
    output_cur = output_conn.cursor()
    output_cur.execute("DROP TABLE IF EXISTS metadata")
    output_cur.execute("DROP TABLE IF EXISTS images")

    # Build schema with optional model_score column
    images_schema = """
CREATE TABLE images (
  image_id INTEGER PRIMARY KEY AUTOINCREMENT,
  generation_info TEXT,
  image_path TEXT UNIQUE,
  status TEXT CHECK(status IN ('saved', 'binned', 'deleted')),
  metadata_id INTEGER NOT NULL,
  seed INTEGER,
  binned_path TEXT,
  binned_timestamp DATETIME"""

    if include_model_scores:
        images_schema += ",\n  model_score REAL DEFAULT NULL"

    images_schema += ",\n  FOREIGN KEY(metadata_id) REFERENCES metadata(metadata_id),\n  UNIQUE(metadata_id, seed)\n);"

    output_cur.executescript(f"""
CREATE TABLE metadata (
  metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
  metadata_content TEXT UNIQUE
);

{images_schema}
""")

    output_cur.close()
    output_conn.close()


def extract_seed(generation_info):
    match = re.search(r'Seed: (\d+)', generation_info)
    return int(match.group(1)) if match else None


def format_metadata_content(metadata_content: str):
    for b in BANNED:
        metadata_content = metadata_content.replace(b, "")
    return metadata_content


def get_metadata(input_cur, output_cur, image_id, retries=0):
    input_cur.execute(
        "SELECT (m.metadata_content) FROM metadata m, images i WHERE i.image_id = ? AND i.metadata_id = m.metadata_id", (image_id, ))

    row = input_cur.fetchone()
    if row is None:
        if retries < 4:
            return get_metadata(input_cur, output_cur, int(image_id) + 1, retries + 1)
        raise Exception("metadata not found")
    output_cur.execute(
        "SELECT metadata_id FROM metadata WHERE metadata_content = ?", (format_metadata_content(row[0]),))
    return output_cur.fetchone()[0]


def get_metadata_id(out_cur, metadata_content):
    out_cur.execute(
        "SELECT metadata_id FROM metadata where metadata_content = (?)", (metadata_content,))
    row = out_cur.fetchone()
    if row == None or row[0] == None:
        return None
    return row[0]


def import_sqlite(input_db_path, output_db_path, include_model_scores=False):
    output_conn = sqlite3.connect(output_db_path)
    output_cur = output_conn.cursor()
    input_conn = sqlite3.connect(input_db_path)
    input_cur = input_conn.cursor()

    input_cur.execute("SELECT * from metadata")

    for metadata in input_cur.fetchall():
        try:
            output_cur.execute(
                "INSERT INTO metadata (metadata_content) VALUES (?)", (format_metadata_content(metadata[1]),))
        except sqlite3.Error as error:
            print("Next, ", error)

    input_cur.execute("SELECT *  from images")

    matched_scores = 0
    total_images = 0

    for image in input_cur.fetchall():
        image_id, generation_info, image_path, status, metadata_id, binned_path, binned_timestamp = image
        total_images += 1

        try:
            if metadata_id != None:
                metadata_key = get_metadata(input_cur, output_cur, image_id)
            else:
                metadata_key = get_metadata(
                    input_cur, output_cur, image_id + 1)
        except:
            print(
                f"Error getting metadata on image: {image_id} and metadata_id: '{metadata_id}'")
            continue

        try:
            seed = extract_seed(generation_info)

            if include_model_scores:
                # Get model score for this image
                model_score = get_model_score(image_path)
                if model_score is not None:
                    matched_scores += 1

                output_cur.execute(
                    "INSERT INTO images (generation_info, image_path, status, binned_path, binned_timestamp, metadata_id, seed, model_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (generation_info, image_path, status, binned_path, binned_timestamp, metadata_key, seed, model_score))
            else:
                output_cur.execute(
                    "INSERT INTO images (generation_info, image_path, status, binned_path, binned_timestamp, metadata_id, seed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (generation_info, image_path, status, binned_path, binned_timestamp, metadata_key, seed))
        except sqlite3.Error as error:
            # TODO: update status
            print("image insertion error, ", error)

    if include_model_scores:
        print(
            f"📊 Model scores: {matched_scores}/{total_images} images matched")

    output_conn.commit()
    output_conn.close()


def import_json(json_file, output_db_path, include_model_scores=False):
    output_conn = sqlite3.connect(output_db_path)
    output_cur = output_conn.cursor()

    prompt_regex = PROMPT_REGEX
    counter = 0
    matched_scores = 0

    with open(json_file, "r") as f:
        data = json.load(f)

    for item in data["deleted"]:

        if not re.match(prompt_regex, item["metadata"]):
            continue

        try:
            metadata_content = "\n".join(item["metadata"].split('\n')[:-1])
            metadata_content = format_metadata_content(metadata_content)
            generation_info = item["metadata"].split('\n')[-1]
        except KeyError as e:
            print(item)
            raise e

        seed = extract_seed(generation_info)
        if seed == None:
            continue
        metadata_id = get_metadata_id(output_cur, metadata_content)
        if metadata_id == None:
            output_cur.execute(
                "INSERT INTO metadata (metadata_content) VALUES (?)", (metadata_content,))
            metadata_id = output_cur.lastrowid

        if include_model_scores:
            # Since JSON imports don't have image_path, we can't match model scores
            # They would need to be added later via update script
            output_cur.execute(
                "INSERT OR IGNORE INTO images (generation_info, status, metadata_id, seed, model_score) VALUES (?,'binned',?,?,?)",
                (generation_info, metadata_id, seed, None))
        else:
            output_cur.execute(
                "INSERT OR IGNORE INTO images (generation_info, status, metadata_id, seed) VALUES (?,'binned',?,?)",
                (generation_info, metadata_id, seed,))

    for item in data["saved"]:
        if not re.match(prompt_regex, item["metadata"]):
            continue

        metadata_content = "\n".join(item["metadata"].split('\n')[:-1])
        metadata_content = format_metadata_content(metadata_content)
        generation_info = item["metadata"].split('\n')[-1]

        seed = extract_seed(generation_info)
        if seed == None:
            continue
        metadata_id = get_metadata_id(output_cur, metadata_content)
        if metadata_id == None:
            output_cur.execute(
                "INSERT INTO metadata (metadata_content) VALUES (?)", (metadata_content,))
            metadata_id = output_cur.lastrowid

        if include_model_scores:
            # Since JSON imports don't have image_path, we can't match model scores
            # They would need to be added later via update script
            output_cur.execute(
                "INSERT OR IGNORE INTO images (generation_info, status, metadata_id, seed, model_score) VALUES (?,'saved',?,?,?)",
                (generation_info, metadata_id, seed, None))
        else:
            output_cur.execute(
                "INSERT OR IGNORE INTO images (generation_info, status, metadata_id, seed) VALUES (?,'saved',?,?)",
                (generation_info, metadata_id, seed,))

    print(f"JSON import completed. Counter: {counter}")
    if include_model_scores:
        print(
            f"📊 Model scores: {matched_scores} matched (JSON imports require separate score update)")

    output_conn.commit()
    output_conn.close()


SD_DB = "./sd.db"


def main():
    parser = argparse.ArgumentParser(
        description="ETL script for image database")
    parser.add_argument(
        "--include-model-scores",
        action="store_true",
        help="Include model_score column and populate from predictions file"
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        help="Path to model predictions JSON file (required if --include-model-scores)"
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default=SD_DB,
        help="Output database path"
    )

    args = parser.parse_args()

    include_scores = args.include_model_scores

    if include_scores:
        if not args.predictions_file:
            print("❌ --predictions-file is required when using --include-model-scores")
            return

        # Load model predictions
        load_model_predictions(args.predictions_file)

    print(f"🔧 Starting ETL process...")
    print(f"   Output database: {args.output_db}")
    print(f"   Include model scores: {include_scores}")
    if include_scores:
        print(f"   Predictions file: {args.predictions_file}")

    # Clean and create database with appropriate schema
    clean_db(args.output_db, include_model_scores=include_scores)

    # Import data
    try:
        import_sqlite("./dbs/data.db", args.output_db,
                      include_model_scores=include_scores)
    except Exception as e:
        print(f"⚠️  Error importing data.db: {e}")

    try:
        import_sqlite("./dbs/data_old.db", args.output_db,
                      include_model_scores=include_scores)
    except Exception as e:
        print(f"⚠️  Error importing data_old.db: {e}")

    try:
        import_json("./save.json", args.output_db,
                    include_model_scores=include_scores)
    except Exception as e:
        print(f"⚠️  Error importing save.json: {e}")

    print("✅ ETL process completed!")


if __name__ == "__main__":
    main()
