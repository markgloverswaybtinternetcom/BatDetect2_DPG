import pandas, os, sys, torch, colorama, json, utils, time, soundfile
from batdetect2.detector.parameters import DEFAULT_MODEL_PATH
from batdetect2.api import load_model, get_config
import batdetect2.utils.detector_utils as du
import batdetect2.utils.audio_utils as au
import batdetect2.detector.compute_features as feats
from batdetect2.types import ( DetectionModel, ProcessingConfiguration, RunResults)
from typing import Any, Union
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_PROB = 0.2

class Classifier():
    """Uses BatDetect2 lower level code without modification any modifications are in this class"""
    def __init__(self): 
        detection_threshold = 0.5; time_expansion_factor = 1; chunk_size = 2.0 # defaults if no config file
        args = {'cnn_features': False, 'spec_features': False, 'quiet': False, 'save_preds_if_empty': False, 'model_path': DEFAULT_MODEL_PATH}
        self.model, params = load_model(DEFAULT_MODEL_PATH) 
        self.config = get_config(**{**params, **args, "time_expansion": time_expansion_factor, "spec_slices": False, "chunk_size": chunk_size, "detection_threshold": detection_threshold})
        code_dir = os.path.dirname(os.path.abspath(__file__))
        speciesNames = pandas.read_csv(os.path.join(code_dir, "Resources", "SpeciesNames.csv"))
        config = None
        configFile = os.path.join(code_dir, "gui_Config.json")
        if os.path.exists(configFile):
            with open(configFile, "r") as jsonfile:
                config = json.load(jsonfile)
                speciesLanguage = config["SpeciesLanguage"] 
        else: speciesLanguage = "EnglishAbbrev"
        if speciesLanguage != 'Latin': self.latinToLangDict = speciesNames.set_index('Latin')[speciesLanguage].to_dict()
        else: self.latinToLangDict = None

    def GetDfSummary(self, df):
        summaryDict = {}
        for row in df.itertuples():
            id = row[6]; prob = float(row[1]) * float(row[7])
            if prob > MIN_PROB:
                if id in summaryDict:
                    min = summaryDict[id][1]; max = summaryDict[id][2];
                    if prob < min: min = prob
                    if prob > max: max = prob
                    summaryDict[id] = [summaryDict[id][0]+1, min, max]
                else:
                    summaryDict[id] = [1, prob, prob]
        summary = ""
        for id, val in summaryDict.items():
            if self.latinToLangDict is None: species = id
            else: species = self.latinToLangDict[id]
            if val[0] == 1: summary += f"{species} 1 call {val[1]:.0%}, "
            else: summary += f"{species} {val[0]} calls {val[2]:.0%}-{val[1]:.0%}, "
        return summary
        
    def save_results_to_file(self, results, op_path: str) -> None:
        summary = ""
        if len(results) > 0:
            result_list = results["pred_dict"]["annotation"] # save csv file - if there are predictions
            results_df = pandas.DataFrame(result_list)
            results_df["file_name"] = results["pred_dict"]["id"] # add file name as a column
            results_df.index.name = "id" # rename index column   
            if "class_prob" in results_df.columns:  # create a csv file with predicted events
                preds_df = results_df[["det_prob", "start_time",  "end_time", "high_freq", "low_freq", "class", "class_prob", "event"]]
                preds_df.to_csv(op_path + ".csv", sep=",")
                summary = self.GetDfSummary(preds_df)
            else:
                with open(op_path + ".csv", "w") as f:
                    f.write("id,det_prob,start_time,end_time,high_freq,low_freq,class,class_prob\n")
        else:
            with open(op_path + ".csv", "w") as f: # empty file so do not repeat classification
                f.write("id,det_prob,start_time,end_time,high_freq,low_freq,class,class_prob\n")
        return summary
    
    def call_pred(self, det_prob, class_prob): 
        classes = list()
        class_calls = list()
        class_max_prob = list()
        class_min_prob = list()
        
        for call in range(det_prob.size):        
            call_best_class = class_prob.argmax(axis=0)[call] #class index of max
            call_best_class_prob = class_prob.max(axis=0)[call] # max call probability
            try:
                i = classes.index(call_best_class) # exception if class does not exist
                class_calls[i] = class_calls[i]+1
                if call_best_class_prob > class_max_prob[i]:
                    class_max_prob[i] = call_best_class_prob
                if call_best_class_prob < class_min_prob[i]:
                    class_min_prob[i] = call_best_class_prob
            except ValueError:  # if class does not exist yet
                classes.append(call_best_class)
                class_calls.append(1)
                class_max_prob.append(call_best_class_prob)
                class_min_prob.append(call_best_class_prob)
        return classes, class_calls , class_max_prob, class_min_prob

    def process_file(self, audio_file: str, model: DetectionModel, config: ProcessingConfiguration, device: torch.device = DEVICE) -> Union[RunResults, Any]:
        predictions = []; spec_feats = []
        info = soundfile.info(audio_file)
        file_samp_rate = info.samplerate
        orig_samp_rate = file_samp_rate * (config.get("time_expansion") or 1)
        sampling_rate, audio_full = au.load_audio( audio_file, time_exp_fact=config.get("time_expansion", 1) or 1,  target_samp_rate=config["target_samp_rate"], scale=config["scale_raw_audio"], max_duration=config.get("max_duration"))

        # loop through larger file and split into chunks
        # TODO: fix so that it overlaps correctly and takes care of duplicate detections at borders
        for chunk_time, audio in du.iterate_over_chunks( audio_full, sampling_rate, config["chunk_size"]):
            pred_nms, features, spec = du._process_audio_array( audio, sampling_rate, model, config, device)
            spec_np = spec.detach().cpu().numpy().squeeze()
            pred_nms["start_times"] += chunk_time
            pred_nms["end_times"] += chunk_time
            predictions.append(pred_nms)

            # extract features - if there are any calls detected
            if pred_nms["det_probs"].shape[0] == 0:
                continue

            spec_feats.append(feats.get_feats(spec_np, pred_nms, config))

        # Merge results from chunks
        predictions, spec_feats, cnn_feats, spec_slices = du._merge_results(predictions, spec_feats, [], [])

        # convert results to a dictionary in the right format
        calls = du.convert_results(file_id=os.path.basename(audio_file), time_exp=config.get("time_expansion", 1) or 1,
            duration=audio_full.shape[0] / float(sampling_rate), params=config, predictions=predictions,
            spec_feats=spec_feats, cnn_feats=[], spec_slices=[], nyquist_freq=orig_samp_rate / 2)
        return calls

    def File(self, filepath, debug=False):
        dir = os.path.dirname(filepath)
        file = os.path.basename(filepath)
        try:
            calls = self.process_file(filepath, self.model, self.config)
        except  Exception as error:
            print(colorama.Fore.RED + f"Classifier process_file {error}" + colorama.Fore.RESET)
            summary = ""; calls = ""
        op_dir = os.path.join(dir,"ann")
        if not os.path.isdir(op_dir): # make directory if it does not exist
            print("Creating directory for annotation files", op_dir)
            os.makedirs(op_dir)
        summary = self.save_results_to_file(calls, os.path.join(op_dir ,file)) # empty file saves trying to classify again
        if len(summary)> 0: print(colorama.Fore.GREEN + f"{file}, {summary}  " + colorama.Fore.RESET, flush=True)
        return summary