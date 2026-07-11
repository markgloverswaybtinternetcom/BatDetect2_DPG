import os, json, polars, glob, argparse, colorama
from Classifier import Classifier

REFERENCE_COLS = [
    "reference_start",
    "reference_end",
    "reference_low",
    "reference_high",
    "reference_class",
    "reference_event",
]

MODEL_COLS = [
    "model_start",
    "model_end",
    "model_low",
    "model_high",
    "model_class",
    "model_event",
]

MATCH_COLS = [
    "model_start", "model_end",
    "model_low", "model_high",
    "model_class", "model_event",
    "reference_start", "reference_end",
    "reference_low", "reference_high",
    "reference_class", "reference_event",
    "time_iou", "frequency_iou", "iou",
]

def normalize_schema(df: pl.DataFrame, required_cols: list[str]) -> pl.DataFrame:
    for col in required_cols:
        if col not in df.columns:
            df = df.with_columns(polars.lit(None).alias(col))
    return df.select(required_cols)

def write_per_model_class_csv(best_matches_all, model_all, reference_all, class_names, model_file_path):
    # Count model calls per class
    model_counts = (model_all.group_by("model_class").count().rename({"count": "model_count"}))
    # Count reference calls per class
    ref_counts = (reference_all.group_by("reference_class").count().rename({"count": "ref_count"}))
    # Count true positives per class
    tp_per_class = (best_matches_all.group_by("model_class").count().rename({"count": "true_positives"}))
    # Merge TP, model_count, ref_count
    per_class = (tp_per_class
        .join(model_counts, left_on="model_class", right_on="model_class", how="left")
        .join(ref_counts, left_on="model_class", right_on="reference_class", how="left"))
    # Compute FP and FN
    per_class = per_class.with_columns([
        (polars.col("model_count") - polars.col("true_positives")).alias("false_positives"),
        (polars.col("ref_count") - polars.col("true_positives")).alias("false_negatives")
    ])
    # Compute precision, recall, F1
    per_class = per_class.with_columns([
        (polars.col("true_positives") /
         (polars.col("true_positives") + polars.col("false_positives")))
         .alias("precision"),

        (polars.col("true_positives") /
         (polars.col("true_positives") + polars.col("false_negatives")))
         .alias("recall"),
    ])
    per_class = per_class.with_columns([
        (polars.col("true_positives") /
         (polars.col("true_positives") + polars.col("false_positives")))
         .alias("precision"),
        (polars.col("true_positives") /
         (polars.col("true_positives") + polars.col("false_negatives")))
         .alias("recall"),
        (2 * polars.col("precision") * polars.col("recall") /
         (polars.col("precision") + polars.col("recall")))
         .alias("f1_score")
    ])
    # Add model name column
    model_name = os.path.basename(model_file_path)
    per_class = per_class.with_columns([polars.lit(model_name).alias("model_name")])
    # Reorder columns
    per_class = per_class.select([
        "model_name",
        "model_class",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1_score",
        "model_count",
        "ref_count"
    ])
    # Write CSV
    out_path = f"{model_name}_class_scores.csv"
    per_class.write_csv(out_path)
    print("Wrote:", out_path)
    
def validate_model(model_file_path, validation_data_directory):
    # Load classifier
    classifier = Classifier(model=model_file_path)
    class_names = classifier.modelParams["class_names"]
    # Collect all WAV files in validation directory
    audio_files = glob.glob(os.path.join(validation_data_directory, "**", "*.wav"), recursive=True)
    all_best_matches = []
    all_model_annotations = []
    all_reference_annotations = []
    for audio_file in audio_files:
        # Run classifier and write model annotation JSON
        classifier.File(audio_file)
        audio_dir = os.path.dirname(audio_file)
        filename = os.path.basename(audio_file)
        model_json_path = os.path.join(audio_dir, "ann", filename + ".json")
        reference_json_path = os.path.join(audio_dir, "valid_ann", filename + ".json")
        # Load JSON
        if not os.path.exists(model_json_path):
            print("Missing model JSON in:", model_json_path)
            continue
        if not os.path.exists(reference_json_path):
            print("Missing reference JSON in:", reference_json_path)
            continue
        with open(model_json_path) as f:
            model_json = json.load(f)
        with open(reference_json_path) as f:
            reference_json = json.load(f)   
        # Convert to Polars DataFrames
        model_df = polars.DataFrame(model_json["annotation"])
        reference_df = polars.DataFrame(reference_json["annotation"])
        # Handle empty JSON (noise or no detections)
        if model_df.is_empty():
            # Create an empty DataFrame with the correct schema
            model_df = polars.DataFrame({
                "model_class": [],
                "model_start": [],
                "model_end": [],
                "model_low": [],
                "model_high": [],
                "model_event": [],
                "class_prob": [],
                "det_prob": [],
                "individual": [],
            })
        else:
            # Normal rename path
            model_df = model_df.rename({
                "class": "model_class",
                "event": "model_event",
                "start_time": "model_start",
                "end_time": "model_end",
                "low_freq": "model_low",
                "high_freq": "model_high",
            })
        if reference_df.is_empty():
            # Create an empty DataFrame with the correct schema
            reference_df = polars.DataFrame({
                "reference_class": [],
                "reference_start": [],
                "reference_end": [],
                "reference_low": [],
                "reference_high": [],
                "reference_event": [],
                "class_prob": [],
                "det_prob": [],
                "individual": [],
            })
        else:
            # Normal rename path
            reference_df = reference_df.rename({
                "class": "reference_class",
                "event": "reference_event",
                "start_time": "reference_start",
                "end_time": "reference_end",
                "low_freq": "reference_low",
                "high_freq": "reference_high"
            })
        # Cross join model × reference calls
        pairs = model_df.join(reference_df, how="cross")
        pairs = pairs.with_columns([
            polars.col("model_start").cast(polars.Float64),
            polars.col("model_end").cast(polars.Float64),
            polars.col("reference_start").cast(polars.Float64),
            polars.col("reference_end").cast(polars.Float64),
            polars.col("model_low").cast(polars.Float64),
            polars.col("model_high").cast(polars.Float64),
            polars.col("reference_low").cast(polars.Float64),
            polars.col("reference_high").cast(polars.Float64),
        ])
        # Compute IoU in Polars
        time_intersection = (
            polars.min_horizontal("model_end", "reference_end") -
            polars.max_horizontal("model_start", "reference_start")
        ).clip(lower_bound=0)
        time_union = (
            polars.max_horizontal("model_end", "reference_end") -
            polars.min_horizontal("model_start", "reference_start")
        )
        pairs = pairs.with_columns([(time_intersection / time_union).alias("time_iou")])         
        freq_intersection = (
            polars.min_horizontal("model_high", "reference_high")
            -
            polars.max_horizontal("model_low", "reference_low")
        ).clip(lower_bound=0)
        freq_union = (
            polars.max_horizontal("model_high", "reference_high") -
            polars.min_horizontal("model_low", "reference_low")
        )
        pairs = pairs.with_columns([
            (freq_intersection / freq_union).alias("frequency_iou")
        ])
        pairs = pairs.with_columns([
            (polars.col("time_iou") * polars.col("frequency_iou")).alias("iou")
        ])        
        # Filter valid matches
        matches = pairs.filter(
            (polars.col("model_class") == polars.col("reference_class")) &
            (polars.col("model_event") == polars.col("reference_event")) &
            (polars.col("iou") >= 0.3)
        )
        # Greedy best match per model call
        best_matches = (
            matches.sort("iou", descending=True)
            .group_by(["model_start", "model_end", "model_low", "model_high"])
            .head(1)
        )
        all_best_matches.append(best_matches)
        all_model_annotations.append(model_df)
        all_reference_annotations.append(reference_df)
    # Concatenate all results
    normalized_refs = [normalize_schema(df, REFERENCE_COLS) for df in all_reference_annotations]
    #initial validation files string floating point values
    normalized_refs = [
        df.with_columns([
            polars.col("reference_start").cast(polars.Float64),
            polars.col("reference_end").cast(polars.Float64),
            polars.col("reference_low").cast(polars.Float64),
            polars.col("reference_high").cast(polars.Float64),
        ])
        for df in normalized_refs
    ]    
    reference_all = polars.concat(normalized_refs)    
    normalized_models = [normalize_schema(df, MODEL_COLS) for df in all_model_annotations]
    model_all = polars.concat(normalized_models)
    normalized_matches = [ normalize_schema(df, MATCH_COLS) for df in all_best_matches]
    best_matches_all = polars.concat(normalized_matches)

    if model_all.is_empty():
        print(colorama.Back.RED + WARNING: No model detections found in validation set." + colorama.Back.RESET)

    if reference_all.is_empty():
        print(colorama.Back.RED + "WARNING: No reference annotations found in validation set." + colorama.Back.RESET)

    if best_matches_all.is_empty():
        print(colorama.Back.RED + "WARNING: No matches found (IoU threshold too high or no overlapping calls)." + colorama.Back.RESET)
    
    print("model_all rows:", model_all.height())
    print("reference_all rows:", reference_all.height())
    print("best_matches_all rows:", best_matches_all.height())

    # Compute per-class CSV
    write_per_model_class_csv(best_matches_all, model_all, reference_all, class_names, model_file_path)
    print("Validation complete.")
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description="Run validation scoring on a trained model.")
    parser.add_argument("validation_data_dir", type=str, help="Path to the root directory of the validation dataset.")
    parser.add_argument("model_path",type=str,help="Path to the trained model file.")
    arguments = parser.parse_args()
    validate_model(model_file_path=arguments.model_path,validation_data_directory=arguments.validation_data_dir)