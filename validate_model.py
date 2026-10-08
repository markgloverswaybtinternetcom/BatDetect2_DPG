import os, polars, glob, argparse, colorama, sys, warnings
from Classifier import Classifier
warnings.filterwarnings("ignore", category=DeprecationWarning)

MIN_IOU_HORSESHOE = 0.05
MIN_IOU_OTHER = 0.15
REFERENCE_COLS = [ "reference_start", "reference_end", "reference_low", "reference_high", "species", "call_type"]
MODEL_COLS = [ "model_start", "model_end", "model_low", "model_high", "model_class", "model_event"]
MATCH_COLS = ["model_start", "model_end", "model_low", "model_high", "model_class", "model_event", 
    "reference_start", "reference_end", "reference_low", "reference_high", "species", "call_type",
    "time_iou", "frequency_iou", "iou"]
EMPTY_MODEL = polars.DataFrame({
    "model_start": polars.Series([], dtype=polars.Float64),
    "model_end": polars.Series([], dtype=polars.Float64),
    "model_low": polars.Series([], dtype=polars.Float64),
    "model_high": polars.Series([], dtype=polars.Float64),
    "model_class": polars.Series([], dtype=polars.Utf8),
    "model_event": polars.Series([], dtype=polars.Utf8)})
EMPTY_REFERENCE = polars.DataFrame({
    "reference_start": polars.Series([], dtype=polars.Float64),
    "reference_end": polars.Series([], dtype=polars.Float64),
    "reference_low": polars.Series([], dtype=polars.Float64),
    "reference_high": polars.Series([], dtype=polars.Float64),
    "species": polars.Series([], dtype=polars.Utf8),
    "call_type": polars.Series([], dtype=polars.Utf8)})

def safe_concat(dfs):
    if not dfs:
        print(colorama.Back.RED + "[SAFE_CONCAT] dfs is empty → returning empty DF"+ colorama.Back.RESET)
        return polars.DataFrame()
    if len(dfs) == 1:
        return dfs[0]
    try:
        return polars.concat(dfs)
    except Exception as e:
        print(colorama.Back.RED + "=== SAFE_CONCAT FAILURE ==="+ colorama.Back.RESET )
        print("Original error:", e)
        print("Concat failed. Dumping schemas:\n")
        for i, df in enumerate(dfs):
            print(f"DF #{i}: {df.columns=} {df.dtypes=} {df.height=}")
        raise
    
def enforce_schema(df, cols):
    # Ensure all required columns exist
    for col in cols:
        if col not in df.columns:
            df = df.with_columns(polars.lit(None).alias(col))
    # Drop extras and reorder
    return df.select(cols)

def cast_numeric(df, numeric_cols):
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(polars.col(col).cast(polars.Float64, strict=False))
    return df
    
def write_per_model_class_csv(best_matches_all, model_all, reference_all, class_names, model_file_path, last=None, max=None, writeFile=True):
    # Build full reference grid (all species × all events)
    ref_species = reference_all["species"].unique().sort()
    ref_events  = reference_all["call_type"].unique().sort()

    full_grid = polars.DataFrame({"species": [s for s in ref_species for _ in ref_events],
        "call_type": [e for _ in ref_species for e in ref_events]})

    # Count model calls per class
    model_counts = (model_all.group_by("model_class", "model_event").count().rename({"count": "model_count"}))
    # Count reference calls per class
    ref_counts = (reference_all.group_by("species", "call_type").count().rename({"count": "ref_count"}))
    # Count true positives per class
    tp_per_class = (best_matches_all.group_by("species", "call_type" ).count().rename({"count": "true_positives"}))
    # Merge TP, model_count, ref_count onto full reference grid
    per_class = (full_grid.join(tp_per_class, on=["species", "call_type"], how="left")
        .join(ref_counts, on=["species", "call_type"], how="left")
        .join(model_counts, left_on=["species", "call_type"], right_on=["model_class", "model_event"], how="left"))
    per_class = per_class.fill_null(0)
    # Compute FP and FN
    per_class = per_class.with_columns([
        (polars.col("model_count").cast(polars.Int64) - polars.col("true_positives")).alias("false_positives"),
        (polars.col("ref_count").cast(polars.Int64) - polars.col("true_positives")).alias("false_negatives")])    
    # Compute precision, recall, F1
    per_class = per_class.with_columns([(polars.col("true_positives") / (polars.col("true_positives") + polars.col("false_positives"))).alias("precision"),
        (polars.col("true_positives") / (polars.col("true_positives") + polars.col("false_negatives"))).alias("recall"),
        (2 * polars.col("true_positives") / (2 * polars.col("true_positives") + polars.col("false_positives") + polars.col("false_negatives"))).alias("f1_score")])
    # Add model name column
    model_name = ""
    if model_file_path is not None:
        model_name = os.path.basename(model_file_path)
    per_class = per_class.with_columns([polars.lit(model_name).alias("model_name")])
    # Reorder columns
    per_class = per_class.select([ "model_name", "species", "call_type", "true_positives", "false_positives", "false_negatives",
        "precision", "recall", "f1_score", "model_count", "ref_count"])
    summary = per_class.select([ polars.lit(model_name).alias("model_name"), polars.lit("ALL").alias("species"), polars.lit("ALL").alias("call_type"),
        polars.sum("true_positives").alias("true_positives"), polars.sum("false_positives").alias("false_positives"), polars.sum("false_negatives").alias("false_negatives"),
        (polars.sum("true_positives") / (polars.sum("true_positives") + polars.sum("false_positives"))).alias("precision"),
        (polars.sum("true_positives") / (polars.sum("true_positives") + polars.sum("false_negatives"))).alias("recall"),
        (2 * polars.sum("true_positives") / (2 * polars.sum("true_positives") +
          polars.sum("false_positives") + polars.sum("false_negatives"))).alias("f1_score"),
        polars.sum("model_count").alias("model_count"), polars.sum("ref_count").alias("ref_count")])
    social_summary = (per_class.filter(polars.col("call_type") == "Social").select([polars.lit(model_name).alias("model_name"), polars.lit("ALL").alias("species"), polars.lit("SOCIAL").alias("call_type"),
        polars.sum("true_positives").alias("true_positives"), polars.sum("false_positives").alias("false_positives"), polars.sum("false_negatives").alias("false_negatives"),
        (polars.sum("true_positives") / (polars.sum("true_positives") + polars.sum("false_positives")) ).alias("precision"),
        (polars.sum("true_positives") / (polars.sum("true_positives") + polars.sum("false_negatives")) ).alias("recall"),
        (2 * polars.sum("true_positives") / (2 * polars.sum("true_positives") + polars.sum("false_positives") + polars.sum("false_negatives")) ).alias("f1_score"),
        polars.sum("model_count").alias("model_count"), polars.sum("ref_count").alias("ref_count"),]))
    buzz_summary = (per_class.filter(polars.col("call_type") == "Feeding Buzz").select([polars.lit(model_name).alias("model_name"), polars.lit("ALL").alias("species"), polars.lit("FEEDING BUZZ").alias("call_type"),
        polars.sum("true_positives").alias("true_positives"), polars.sum("false_positives").alias("false_positives"), polars.sum("false_negatives").alias("false_negatives"),
        (polars.sum("true_positives") / (polars.sum("true_positives") + polars.sum("false_positives")) ).alias("precision"),
        (polars.sum("true_positives") / (polars.sum("true_positives") + polars.sum("false_negatives")) ).alias("recall"),
        (2 * polars.sum("true_positives") / (2 * polars.sum("true_positives") + polars.sum("false_positives") + polars.sum("false_negatives")) ).alias("f1_score"),
        polars.sum("model_count").alias("model_count"), polars.sum("ref_count").alias("ref_count"),]))
    echolocation_summary = (per_class.filter(polars.col("call_type") == "Echolocation").select([polars.lit(model_name).alias("model_name"), polars.lit("ALL").alias("species"), polars.lit("ECHOLOCATION").alias("call_type"),
        polars.sum("true_positives").alias("true_positives"), polars.sum("false_positives").alias("false_positives"), polars.sum("false_negatives").alias("false_negatives"),
        (polars.sum("true_positives") / (polars.sum("true_positives") + polars.sum("false_positives")) ).alias("precision"),
        (polars.sum("true_positives") / (polars.sum("true_positives") + polars.sum("false_negatives")) ).alias("recall"),
        (2 * polars.sum("true_positives") / (2 * polars.sum("true_positives") + polars.sum("false_positives") + polars.sum("false_negatives")) ).alias("f1_score"),
        polars.sum("model_count").alias("model_count"), polars.sum("ref_count").alias("ref_count"),]))
    per_class = per_class.sort(["species", "call_type"])
    per_class = polars.concat([per_class, echolocation_summary, social_summary, buzz_summary, summary])
    # Round all float columns to the desired precision
    DECIMALS = 3
    per_class = per_class.select([
        polars.col(col).round(DECIMALS) if per_class[col].dtype in (polars.Float32, polars.Float64) else polars.col(col)
        for col in per_class.columns
    ])
    if writeFile:
        # Write CSV
        model_dir = os.path.dirname(model_file_path)
        out_path = os.path.join(model_dir, f"{model_name}_class_scores.csv")
        per_class.write_csv(out_path)
        print("Wrote:", out_path)
    f1_score = summary.get_column("f1_score").sum(); 
    echolocation = echolocation_summary.get_column("f1_score").sum();
    social = social_summary.get_column("f1_score").sum(); 
    buzz = buzz_summary.get_column("f1_score").sum(); 
    true_positives = summary.get_column("true_positives").sum(); false_positives = summary.get_column("false_positives").sum(); false_negatives = summary.get_column("false_negatives").sum()
    F1_Score = f"{colorama.Style.BRIGHT}{f1_score=:.4f}{colorama.Style.NORMAL}"
    if last is not None and f1_score < last:       
        print(colorama.Fore.RED + f"{model_name} {true_positives=}, {false_positives=}, {false_negatives=}, " + F1_Score + f", {echolocation=:.4f}, {social=:.4f} {buzz=:.4f}"+ colorama.Fore.RESET)
    else:
        if max is not None and f1_score >= max:
            print(colorama.Fore.GREEN + f"{model_name} {true_positives=}, {false_positives=}, {false_negatives=}, " + F1_Score + f", {echolocation=:.4f}, {social=:.4f} {buzz=:.4f}"+ colorama.Fore.RESET)
        else:
            print(colorama.Fore.YELLOW + f"{model_name} {true_positives=}, {false_positives=}, {false_negatives=} " + F1_Score + f", {echolocation=:.4f}, {social=:.4f} {buzz=:.4f}"+ colorama.Fore.RESET)
    return f1_score

def latest_model_file(models_dir):
    files = glob.glob(os.path.join(models_dir, "model_*.pth.tar"))
    if not files:
        raise FileNotFoundError("No model files found in Models/")
    # Sort by model number, then epoch
    def extract_nums(f):
        base = os.path.basename(f)
        parts = base.split("_")
        model_num = int(parts[1])
        epoch = int(parts[2][1:].split(".")[0])  # strip leading E
        return (model_num, epoch)
    files.sort(key=extract_nums)
    return files[-1]

def validate_model(model_file_path, validation_data_directory, last=None, max=None, writeFile=True):
    # Load classifier
    classifier = Classifier(modelPath=model_file_path, debug=False)
    class_names = classifier.modelParams["class_names"]
    # Collect all WAV files in validation directory
    audio_files = glob.glob(os.path.join(validation_data_directory, "**", "*.wav"), recursive=True)
    all_best_matches = []
    all_model_annotations = []
    all_reference_annotations = []
    all_file_summaries = []
    for audio_file in audio_files:
        # Run classifier and write model annotation CSV
        classifier.File(audio_file, printSummary=False)
        audio_dir = os.path.basename(os.path.dirname(audio_file))
        filename = os.path.basename(audio_file)
        model_csv_path = os.path.join(validation_data_directory, audio_dir, "ann", filename + ".csv")
        reference_csv_path = os.path.join(validation_data_directory, audio_dir, "valid_ann", filename + ".csv")
        # Load CSV
        if not os.path.exists(model_csv_path):
            print("Missing model annotation CSV in:", model_csv_path)
            continue
        if not os.path.exists(reference_csv_path):
            print(colorama.Back.RED + f"Missing reference CSV in: {reference_csv_path}" + colorama.Back.RESET)
            continue
        model_df = polars.read_csv(model_csv_path)
        reference_df = polars.read_csv(reference_csv_path) 
        # Handle empty CSV (noise or no detections)
        if model_df.is_empty():
            model_df = EMPTY_MODEL
        else:
            # Normal rename path
            model_df = model_df.rename({"class": "model_class","event": "model_event", "start_time": "model_start",
                "end_time": "model_end", "low_freq": "model_low", "high_freq": "model_high"})
        model_df = enforce_schema(model_df, MODEL_COLS)
        model_df = cast_numeric(model_df, ["model_start", "model_end", "model_low", "model_high"]) # still needed as freq is INT64
        if reference_df.is_empty():
            reference_df = EMPTY_REFERENCE 
        else:
            reference_df = reference_df.rename({"class": "species","event": "call_type","start_time": "reference_start",
                "end_time": "reference_end","low_freq": "reference_low","high_freq": "reference_high"})  
        reference_df = enforce_schema(reference_df, REFERENCE_COLS)
        reference_df = cast_numeric(reference_df, ["reference_start", "reference_end", "reference_low", "reference_high"])  # still needed as freq is INT64
        all_reference_annotations.append(reference_df)
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
            polars.min_horizontal("model_high", "reference_high") - polars.max_horizontal("model_low", "reference_low")).clip(lower_bound=0)
        freq_union = (polars.max_horizontal("model_high", "reference_high") - polars.min_horizontal("model_low", "reference_low"))
        pairs = pairs.with_columns([ (freq_intersection / freq_union).alias("frequency_iou") ])
        pairs = pairs.with_columns([ (polars.col("time_iou") * polars.col("frequency_iou")).alias("iou")]) 
        # Filter valid matches
        
        matches = pairs.filter ( (polars.col("model_class") == polars.col("species")) & (polars.col("model_event") == polars.col("call_type")) &
            (polars.when(polars.col("species") == "Rhinolophus ferrumequinum").then(polars.col("iou") >= MIN_IOU_HORSESHOE).otherwise(polars.col("iou") >= MIN_IOU_OTHER)) )

        #matches = pairs.filter((polars.col("model_class") == polars.col("species")) & (polars.col("model_event") == polars.col("call_type")) & (polars.col("iou") >= MIN_IOU))           
        matches = matches.with_columns(polars.concat_str(["model_start","model_end","model_low","model_high",
            "reference_start","reference_end","reference_low","reference_high"]).alias("pair_key"))
        matches = matches.unique(subset=["pair_key"])
        
        best_matches = (matches.sort("iou", descending=True).group_by(["model_start", "model_end", "model_low", "model_high"]).head(1))
        
        #best_matches = (matches.sort("iou", descending=True).group_by("pair_key").head(1))  
        best_matches = best_matches.drop("pair_key")
        all_best_matches.append(best_matches)
        all_model_annotations.append(model_df)

        if writeFile:
            # --- Per-file CSV row (reconstructed) ---
            # Count model detections for this file
            model_count_file = model_df.shape[0]
            # Count reference annotations for this file
            ref_count_file = reference_df.shape[0]
            # Count true positives for this file
            tp_count_file = best_matches.shape[0]
            # False positives = model detections not matched
            fp_count_file = model_count_file - tp_count_file
            # False negatives = reference annotations not matched
            fn_count_file = ref_count_file - tp_count_file
            # Build per-file row
            file_row = polars.DataFrame({"dir": [audio_dir], "file": [filename],  "true_positives": [tp_count_file], "false_positives": [fp_count_file], "false_negatives": [fn_count_file], "model_count": [model_count_file], "ref_count": [ref_count_file]})
            # Append to list
            all_file_summaries.append(file_row)
            # Concatenate per-file summaries
            file_summary_df = safe_concat(all_file_summaries)
    # Concatenate all results
    reference_all = safe_concat(all_reference_annotations)  
    model_all = safe_concat(all_model_annotations)
    best_matches_all = safe_concat(all_best_matches)
    if model_all.is_empty():
        print(colorama.Back.RED + "WARNING: No model detections found in validation set." + colorama.Back.RESET)
    if reference_all.is_empty():
        print(colorama.Back.RED + "WARNING: No reference annotations found in validation set." + colorama.Back.RESET)
    if best_matches_all.is_empty():
        print(colorama.Back.RED + "WARNING: No matches found (IoU threshold too high or no overlapping calls)." + colorama.Back.RESET)
    if writeFile:
        # Write per-file CSV
        out_path = os.path.join(os.path.dirname(model_file_path), f"{os.path.basename(model_file_path)}_per_file_scores.csv")
        file_summary_df.write_csv(out_path)
        print("Wrote per-file CSV:", out_path)
    # Compute per-class CSV
    f1_score = write_per_model_class_csv(best_matches_all, model_all, reference_all, class_names, model_file_path, last, max, writeFile)
    return f1_score
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description="Run validation scoring on a trained model.")
    parser.add_argument("validation_data_dir", type=str, help="Path to the root directory of the validation dataset.")
    parser.add_argument("model_dir",type=str,help="Directory containing trained model files.")
    arguments = parser.parse_args()
    if os.path.isdir(arguments.model_dir):
        model_file_path = latest_model_file(arguments.model_dir)
    elif arguments.model_dir.endswith(".pth.tar"): model_file_path = arguments.model_dir
    else: sys.exit(colorama.Back.RED +  f"Invalid model path {arguments.model_dir}" + colorama.Back.RESET) 
    validate_model(model_file_path, validation_data_directory=arguments.validation_data_dir)