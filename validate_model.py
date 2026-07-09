import os, json, polars, glob, argparse

def validate_model(model_file_path, validation_data_directory):
    # Load classifier
    classifier = Classifier(model=model_file_path)
    class_names = classifier.modelParams["class_names"]
    # Prepare output directory for model annotations
    model_annotation_directory = os.path.join(validation_data_directory, "model_annotations")
    os.makedirs(model_annotation_directory, exist_ok=True)
    # Collect all WAV files in validation directory
    audio_files = glob.glob(os.path.join(validation_data_directory, "**", "*.wav"), recursive=True)
    all_best_matches = []
    all_model_annotations = []
    all_reference_annotations = []
    for audio_file in audio_files:
        # Run classifier and write model annotation JSON
        classifier.File(audio_file, annDir=model_annotation_directory)
        base_name = os.path.basename(audio_file)
        model_json_path = os.path.join(os.path.dirname(audio_file), "model_annotations", base_name + ".json")
        reference_json_path = os.path.join(os.path.dirname(audio_file), "reference_annotations", base_name + ".json")
        # Load JSON
        with open(model_json_path) as f:
            model_json = json.load(f)
        with open(reference_json_path) as f:
            reference_json = json.load(f)
        # Convert to Polars DataFrames
        model_df = polars.DataFrame(model_json["annotation"])
        reference_df = polars.DataFrame(reference_json["annotation"])
        # Rename columns for clarity
        model_df = model_df.rename({
            "class": "model_class",
            "event": "model_event",
            "start_time": "model_start",
            "end_time": "model_end",
            "low_freq": "model_low",
            "high_freq": "model_high"
        })
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
        # Compute IoU in Polars
        time_intersection = (
            polars.min_horizontal("model_end", "reference_end") -
            polars.max_horizontal("model_start", "reference_start")
        ).clip(lower_bound=0)
        time_union = (
            polars.max_horizontal("model_end", "reference_end") -
            polars.min_horizontal("model_start", "reference_start")
        )
        frequency_intersection = (
            polars.min_horizontal("model_high", "reference_high") -
            polars.max_horizontal("model_low", "reference_low")
        ).clip(lower_bound=0)
        frequency_union = (
            polars.max_horizontal("model_high", "reference_high") -
            polars.min_horizontal("model_low", "reference_low")
        )
        pairs = pairs.with_columns([
            (time_intersection / time_union).alias("time_iou"),
            (frequency_intersection / frequency_union).alias("frequency_iou"),
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
    best_matches_all = polars.concat(all_best_matches)
    model_all = polars.concat(all_model_annotations)
    reference_all = polars.concat(all_reference_annotations)
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