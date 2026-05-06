import pandas, os, sys, torch, colorama, json, utils, time, soundfile, Net2dFast, librosa, librosa.core.spectrum, numpy
from typing import Any, Union, Protocol, TypedDict
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_PROB = 0.2
DEFAULT_MODEL_PATH = "Net2DFast_UK_same.pth.tar"

TARGET_SAMPLERATE_HZ = 256000
FFT_WIN_LENGTH_S = 512 / 256000.0
FFT_OVERLAP = 0.75
MAX_FREQ_HZ = 120000
MIN_FREQ_HZ = 10000
RESIZE_FACTOR = 0.5
SPEC_DIVIDE_FACTOR = 32
SPEC_HEIGHT = 256
DETECTION_THRESHOLD = 0.5
NMS_KERNEL_SIZE = 9
NMS_TOP_K_PER_SEC = 200
SPEC_SCALE = "pcen"
DENOISE_SPEC_AVG = True
MAX_SCALE_SPEC = False
CHUNK_SIZE = 2.0
### types

class DetectionModel(Protocol):
    num_classes: int
    emb_dim: int
    num_filts: int
    resize_factor: float
    ip_height_rs: int

class RunResults(TypedDict):
    pred_dict: FileAnnotations
    spec_feats: NotRequired[List[np.ndarray]]
    spec_feat_names: NotRequired[List[str]]
    cnn_feats: NotRequired[List[np.ndarray]]
    cnn_feat_names: NotRequired[List[str]]
    spec_slices: NotRequired[List[np.ndarray]]
    
def load_model(model_path: str = DEFAULT_MODEL_PATH, load_weights: bool = True, device: Optional[torch.device] = None, weights_only: bool = True) -> Tuple[DetectionModel, ModelParameters]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model file not found.")
    net_params = torch.load(model_path, map_location=device, weights_only=weights_only)

    params = net_params["params"]
    model: DetectionModel
    model = Net2dFast.Net2dFast(params["num_filters"], num_classes=len(params["class_names"]), ip_height=params["ip_height"])

    if load_weights:
        model.load_state_dict(net_params["state_dict"])

    model = model.to(device)
    model.eval()
    return model, params
    
def x_coord_to_sample(x_pos: int) -> int:
    n_fft = numpy.floor(FFT_WIN_LENGTH_S * TARGET_SAMPLERATE_HZ)
    n_overlap = numpy.floor(FFT_OVERLAP * n_fft)
    n_step = n_fft - n_overlap
    x_pos = int(x_pos / RESIZE_FACTOR)
    return int((x_pos * n_step) + n_overlap)
    
def pad_audio(audio: numpy.ndarray, samplerate: int = TARGET_SAMPLERATE_HZ, window_duration: float = FFT_WIN_LENGTH_S,
    window_overlap: float = FFT_OVERLAP, resize_factor: float = RESIZE_FACTOR, divide_factor: int = SPEC_DIVIDE_FACTOR, fixed_width: Optional[int] = None):
    spec_width = compute_spectrogram_width(audio.shape[0])

    if fixed_width:
        target_samples = x_coord_to_sample(fixed_width)

        if spec_width < fixed_width:
            # need to be at least min_size
            diff = target_samples - audio.shape[0]
            return numpy.hstack((audio, numpy.zeros(diff, dtype=audio.dtype)))
        if spec_width > fixed_width:
            return audio[:target_samples]
        return audio

    min_width = int(divide_factor / resize_factor)
    if spec_width < min_width:
        target_samples = x_coord_to_sample(min_width)
        diff = target_samples - audio.shape[0]
        return numpy.hstack((audio, numpy.zeros(diff, dtype=audio.dtype)))

    if (spec_width % divide_factor) == 0:
        return audio

    target_width = int(numpy.ceil(spec_width / divide_factor)) * divide_factor
    target_samples = x_coord_to_sample(target_width)
    diff = target_samples - audio.shape[0]
    return numpy.hstack((audio, numpy.zeros(diff, dtype=audio.dtype)))

def gen_mag_spectrogram(x, fs, ms, overlap_perc):
    # Computes magnitude spectrogram by specifying time.
    x = x.astype(numpy.float32)
    nfft = int(ms * fs)
    noverlap = int(overlap_perc * nfft)
    # window data
    step = nfft - noverlap
    # compute spec
    spec, _ = librosa.core.spectrum._spectrogram(y=x, power=1, n_fft=nfft, hop_length=step, center=False)
    # remove DC component and flip vertical orientation
    spec = numpy.flipud(spec[1:, :])
    return spec.astype(numpy.float32)

def generate_spectrogram(audio, sampling_rate, return_spec_for_viz=False, check_spec_size=True):
    # generate spectrogram
    spec = gen_mag_spectrogram(audio, sampling_rate, FFT_WIN_LENGTH_S, FFT_OVERLAP)
    # crop to min/max freq
    max_freq = round(MAX_FREQ_HZ * FFT_WIN_LENGTH_S)
    min_freq = round(MIN_FREQ_HZ * FFT_WIN_LENGTH_S)
    if spec.shape[0] < max_freq:
        freq_pad = max_freq - spec.shape[0]
        spec = numpy.vstack((numpy.zeros((freq_pad, spec.shape[1]), dtype=spec.dtype), spec))
    spec_cropped = spec[-max_freq : spec.shape[0] - min_freq, :]
    spec = librosa.pcen(spec_cropped * (2**31), sr=sampling_rate / 10).astype(numpy.float32)
    spec = spec - numpy.mean(spec, 1)[:, numpy.newaxis]
    spec.clip(min=0, out=spec)

    if return_spec_for_viz:
        log_scaling = (2.0 * (1.0 / sampling_rate) * (1.0 / (numpy.abs(numpy.hanning(int(FFT_WIN_LENGTH_S * sampling_rate))) ** 2).sum()))
        spec_for_viz = numpy.log1p(log_scaling * spec_cropped).astype(numpy.float32)
    else:
        spec_for_viz = None
    return spec, spec_for_viz

def compute_spectrogram_width(length: int) -> int:
    n_fft = int(FFT_WIN_LENGTH_S * TARGET_SAMPLERATE_HZ)
    n_overlap = int(FFT_OVERLAP * n_fft)
    n_step = n_fft - n_overlap
    width = (length - n_overlap) // n_step
    return int(width * RESIZE_FACTOR)

def compute_spectrogram(audio: numpy.ndarray, sampling_rate: int, device: torch.device, return_np: bool = False) -> Tuple[float, torch.Tensor, Optional[numpy.ndarray]]:
    # pad audio so it is evenly divisible by downsampling factors
    duration = audio.shape[0] / float(sampling_rate)
    audio = pad_audio(audio, sampling_rate, FFT_WIN_LENGTH_S, FFT_OVERLAP, RESIZE_FACTOR, SPEC_DIVIDE_FACTOR)
    # generate spectrogram
    spec, _ = generate_spectrogram(audio, sampling_rate)
    # convert to pytorch
    spec = torch.from_numpy(spec).to(device)
    # add batch and channel dimensions
    spec = spec.unsqueeze(0).unsqueeze(0)
    # resize the spec
    resize_factor = RESIZE_FACTOR
    spec_op_shape = (int(SPEC_HEIGHT * resize_factor), int(spec.shape[-1] * resize_factor))
    spec = torch.nn.functional.interpolate(spec, size=spec_op_shape,  mode="bilinear", align_corners=False)
    if return_np: spec_np = spec[0, 0, :].cpu().data.numpy()
    else:  spec_np = None
    return duration, spec, spec_np

    ### detector_utils
def iterate_over_chunks(audio: numpy.ndarray, samplerate: int, chunk_size: float) -> Iterator[Tuple[float, numpy.ndarray]]:
    nsamples = audio.shape[0]
    duration_full = nsamples / samplerate
    num_chunks = int(numpy.ceil(duration_full / chunk_size))
    for chunk_id in range(num_chunks):
        chunk_start = chunk_size * chunk_id
        chunk_length = int(samplerate * chunk_size)
        start_sample = chunk_id * chunk_length
        end_sample = numpy.minimum((chunk_id + 1) * chunk_length, nsamples)
        yield chunk_start, audio[start_sample:end_sample]
        
############################ detector.post_process###########################
def x_coords_to_time(x_pos: float, sampling_rate: int, fft_win_length: float, fft_overlap: float) -> float:
    nfft = int(fft_win_length * sampling_rate)
    noverlap = int(fft_overlap * nfft)
    return ((x_pos * (nfft - noverlap)) + noverlap) / sampling_rate
    
def non_max_suppression(heat: torch.Tensor, kernel_size: Union[int, Tuple[int, int]]):
    # kernel can be an int or list/tuple
    if isinstance(kernel_size, int):
        kernel_size_h = kernel_size
        kernel_size_w = kernel_size
    else:
        kernel_size_h, kernel_size_w = kernel_size
    pad_h = (kernel_size_h - 1) // 2
    pad_w = (kernel_size_w - 1) // 2
    hmax = torch.nn.functional.max_pool2d(heat, (kernel_size_h, kernel_size_w), stride=1, padding=(pad_h, pad_w))
    keep = (hmax == heat).float()
    return heat * keep
    
def get_topk_scores(scores, K):
    # expects input of size:  batch x 1 x height x width
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode="floor").long()
    topk_xs = (topk_inds % width).long()
    return topk_scores, topk_ys, topk_xs

def run_nms(outputs: ModelOutput, sampling_rate: numpy.ndarray) -> Tuple[List[PredictionResults], List[numpy.ndarray]]:
    """Run non-maximum suppression on the output of the model.
    Model outputs processed are expected to have a batch dimension.
    Each element of the batch is processed independently. 
    The result is a pair of lists, one for the predictions and one for the features"""
    pred_det, pred_size, pred_class, _, features = outputs
    pred_det_nms = non_max_suppression(pred_det, NMS_KERNEL_SIZE)
    freq_rescale = (MAX_FREQ_HZ - MIN_FREQ_HZ) / pred_det.shape[-2]
    duration = x_coords_to_time(pred_det.shape[-1], int(sampling_rate[0].item()), FFT_WIN_LENGTH_S, FFT_OVERLAP)
    top_k = int(duration * NMS_TOP_K_PER_SEC)
    scores, y_pos, x_pos = get_topk_scores(pred_det_nms, top_k)
    # loop over batch to save outputs
    preds: List[PredictionResults] = []
    feats: List[numpy.ndarray] = []
    for num_detection in range(pred_det_nms.shape[0]):
        # get valid indices
        inds_ord = torch.argsort(x_pos[num_detection, :])
        valid_inds = (scores[num_detection, inds_ord] > DETECTION_THRESHOLD)
        valid_inds = inds_ord[valid_inds]

        # create result dictionary
        pred = {}
        pred["det_probs"] = scores[num_detection, valid_inds]
        pred["x_pos"] = x_pos[num_detection, valid_inds]
        pred["y_pos"] = y_pos[num_detection, valid_inds]
        pred["bb_width"] = pred_size[num_detection, 0, pred["y_pos"], pred["x_pos"]]
        pred["bb_height"] = pred_size[num_detection, 1, pred["y_pos"], pred["x_pos"]]
        pred["start_times"] = x_coords_to_time(pred["x_pos"].float() / RESIZE_FACTOR, int(sampling_rate[num_detection].item()), FFT_WIN_LENGTH_S, FFT_OVERLAP)
        pred["end_times"] = x_coords_to_time((pred["x_pos"].float() + pred["bb_width"]) / RESIZE_FACTOR,
            int(sampling_rate[num_detection].item()), FFT_WIN_LENGTH_S, FFT_OVERLAP)
        pred["low_freqs"] = (pred_size[num_detection].shape[1] - pred["y_pos"].float()) * freq_rescale + MIN_FREQ_HZ
        pred["high_freqs"] = (pred["low_freqs"] + pred["bb_height"] * freq_rescale)

        # extract the per class votes
        if pred_class is not None:
            pred["class_probs"] = pred_class[num_detection, :, y_pos[num_detection, valid_inds], x_pos[num_detection, valid_inds],]
        # extract the model features
        if features is not None:
            feat = features[num_detection, :, y_pos[num_detection, valid_inds], x_pos[num_detection, valid_inds], ].transpose(0, 1)
            feat = feat.detach().cpu().numpy().astype(numpy.float32)
            feats.append(feat)
        # convert to numpy
        for key, value in pred.items():
            pred[key] = value.detach().cpu().numpy().astype(numpy.float32)
        preds.append(pred)  # type: ignore

    return preds, feats


def _process_spectrogram(spec: torch.Tensor, samplerate: int, model: DetectionModel, modelParams) -> Tuple[PredictionResults, numpy.ndarray]:
    # evaluate model
    with torch.no_grad():
        outputs = model(spec)

    # run non-max suppression
    pred_nms_list, features = run_nms(outputs, numpy.array([float(samplerate)]))
    pred_nms = pred_nms_list[0]

    # if we have a background class
    class_probs = pred_nms.get("class_probs")
    if (class_probs is not None) and (class_probs.shape[0] > len(modelParams["class_names"])):
        pred_nms["class_probs"] = class_probs[:-1, :]

    return pred_nms, numpy.concatenate(features, axis=0)
    
def _process_audio_array(audio: numpy.ndarray, sampling_rate: int, model: DetectionModel,  modelParams, device: torch.device) -> Tuple[PredictionResults, numpy.ndarray, torch.Tensor]:
    # load audio file and compute spectrogram
    _, spec, _ = compute_spectrogram(audio, sampling_rate, device, return_np=False)
    pred_nms, features = _process_spectrogram(spec, sampling_rate, model, modelParams)
    return pred_nms, features, spec 
    
def _merge_results(predictions):
    predictions_m = {
        "det_probs": numpy.array([]), "x_pos": numpy.array([]), "y_pos": numpy.array([]), "bb_widths": numpy.array([]),
        "bb_heights": numpy.array([]), "start_times": numpy.array([]), "end_times": numpy.array([]),
        "low_freqs": numpy.array([]), "high_freqs": numpy.array([]), "class_probs": numpy.array([])}
    num_preds = numpy.sum([len(pp["det_probs"]) for pp in predictions])
    if num_preds > 0:
        for key in predictions[0].keys():
            predictions_m[key] = numpy.hstack([pp[key] for pp in predictions if pp["det_probs"].shape[0] > 0])
    return predictions_m
    
def get_annotations_from_preds(predictions: PredictionResults, class_names: List[str]) -> List[Annotation]:
    """Get list of annotations from predictions."""
    # Get the best class prediction probability and index for each detection
    class_prob_best = predictions["class_probs"].max(0)
    class_ind_best = predictions["class_probs"].argmax(0)
    # Pack the results into a list of dictionaries
    annotations: List[Annotation] = [{
        "start_time": round(float(start_time), 4),
        "end_time": round(float(end_time), 4),
        "low_freq": int(low_freq),
        "high_freq": int(high_freq),
        "class": str(class_names[class_index]),
        "class_prob": round(float(class_prob), 3),
        "det_prob": round(float(det_prob), 3),
        "individual": "-1",
        "event": "Echolocation"}
        for (start_time, end_time, low_freq, high_freq, class_index, class_prob, det_prob) in zip(
            predictions["start_times"], predictions["end_times"], predictions["low_freqs"], predictions["high_freqs"], class_ind_best, class_prob_best, predictions["det_probs"])
    ]
    return annotations

def overall_class_pred(det_prob, class_prob):
    weighted_pred = (class_prob * det_prob).sum(1)
    return weighted_pred / weighted_pred.sum()
    
def format_single_result(file_id: str, time_exp: float, duration: float, predictions: PredictionResults, class_names: List[str]) -> FileAnnotations:
    try:
        # Get a single class prediction for the file
        class_overall = overall_class_pred(predictions["det_probs"], predictions["class_probs"])
        class_name = class_names[numpy.argmax(class_overall)]
        annotations = get_annotations_from_preds(predictions, class_names)
    except (numpy.exceptions.AxisError, ValueError):
        # No detections
        class_overall = numpy.zeros(len(class_names))
        class_name = "None"
        annotations = []

    return {"id": file_id, "annotated": False, "issues": False, "notes": "Automatically generated.", "time_exp": time_exp, "duration": round(float(duration), 4),
        "annotation": annotations, "class_name": class_name}
    
def convert_results(file_id: str, time_exp: float, duration: float, params: ResultParams, predictions, nyquist_freq: Optional[float] = None) -> RunResults:

    pred_dict = format_single_result(file_id, time_exp, duration, predictions, params["class_names"])

    # Remove high frequency detections
    if nyquist_freq is not None:
        pred_dict["annotation"] = [pred for pred in pred_dict["annotation"] if pred["high_freq"] <= nyquist_freq]

    # combine into final results dictionary
    results: RunResults = {"pred_dict": pred_dict}
    return results

    
    ### audio_utils
def load_audio(path: AudioPath, time_exp_fact: float, target_samp_rate: int) -> Tuple[int, numpy.ndarray ]:
    audio_raw, file_sampling_rate = librosa.load(path, sr=None, dtype=numpy.float32)
    
    if len(audio_raw.shape) > 1:
        raise ValueError("Currently does not handle stereo files")

    sampling_rate = file_sampling_rate * time_exp_fact

    # resample - need to do this after correcting for time expansion
    sampling_rate_old = sampling_rate
    sampling_rate = target_samp_rate
    if sampling_rate_old != sampling_rate:
        audio_raw = librosa.resample(audio_raw, orig_sr=sampling_rate_old, target_sr=sampling_rate, res_type="polyphase")
    return sampling_rate, audio_raw

    
    ### detector.compute_features
    
def compute_duration(prediction: types.Prediction, **_,) -> float:
    return round(prediction["end_time"] - prediction["start_time"], 5)

def compute_low_freq(prediction: types.Prediction, **_) -> float:
    return int(prediction["low_freq"])

def compute_high_freq(prediction: types.Prediction, **_) -> float:
    return int(prediction["high_freq"])

def compute_bandwidth(prediction: types.Prediction, **_) -> float:
    return int(prediction["high_freq"] - prediction["low_freq"])

def convert_int_to_freq(spec_ind, spec_height, min_freq, max_freq):
    spec_ind = spec_height - spec_ind
    return round((spec_ind / float(spec_height)) * (max_freq - min_freq) + min_freq, 2)
    
def compute_max_power_bb(prediction: types.Prediction, spec: Optional[np.ndarray] = None, min_freq: int = MIN_FREQ_HZ,  max_freq: int = MAX_FREQ_HZ, **_,) -> float:
    if spec is None:
        return numpy.nan
    x_start = max(0, prediction["x_pos"])
    x_end = min(spec.shape[1] - 1, prediction["x_pos"] + prediction["bb_width"])
    # y low is the lowest freq but it will have a higher value due to array
    # starting at 0 at top
    y_low = min(spec.shape[0] - 1, prediction["y_pos"])
    y_high = max(0, prediction["y_pos"] - prediction["bb_height"])
    spec_bb = spec[y_high:y_low, x_start:x_end]
    power_per_freq_band = numpy.sum(spec_bb, axis=1)
    try:
        max_power_ind = numpy.argmax(power_per_freq_band)
    except ValueError:
        # If the call is too short, the bounding box might be empty.
        # In this case, return NaN.
        return numpy.nan
    return int(convert_int_to_freq(y_high + max_power_ind, spec.shape[0], min_freq, max_freq))

def compute_max_power(prediction: types.Prediction, spec: Optional[np.ndarray] = None, min_freq: int = MIN_FREQ_HZ, max_freq: int = MAX_FREQ_HZ, **_) -> float:
    if spec is None:
        return numpy.nan
    x_start = max(0, prediction["x_pos"])
    x_end = min(spec.shape[1] - 1, prediction["x_pos"] + prediction["bb_width"])
    spec_call = spec[:, x_start:x_end]
    power_per_freq_band = numpy.sum(spec_call, axis=1)
    max_power_ind = numpy.argmax(power_per_freq_band)
    return int(convert_int_to_freq(max_power_ind, spec.shape[0], min_freq, max_freq))

def compute_max_power_first(prediction: types.Prediction, spec: Optional[np.ndarray] = None, min_freq: int = MIN_FREQ_HZ, max_freq: int = MAX_FREQ_HZ, **_) -> float:
    if spec is None:
        return numpy.nan
    x_start = max(0, prediction["x_pos"])
    x_end = min(spec.shape[1] - 1, prediction["x_pos"] + prediction["bb_width"])
    spec_call = spec[:, x_start:x_end]
    first_half = spec_call[:, : int(spec_call.shape[1] / 2)]
    power_per_freq_band = numpy.sum(first_half, axis=1)
    max_power_ind = numpy.argmax(power_per_freq_band)
    return int(convert_int_to_freq(max_power_ind, spec.shape[0], min_freq, max_freq))


def compute_max_power_second(prediction: types.Prediction, spec: Optional[np.ndarray] = None, min_freq: int = MIN_FREQ_HZ, max_freq: int = MAX_FREQ_HZ, **_) -> float:
    if spec is None:
        return numpy.nan
    x_start = max(0, prediction["x_pos"])
    x_end = min(spec.shape[1] - 1, prediction["x_pos"] + prediction["bb_width"])
    spec_call = spec[:, x_start:x_end]
    second_half = spec_call[:, int(spec_call.shape[1] / 2) :]
    power_per_freq_band = numpy.sum(second_half, axis=1)
    max_power_ind = numpy.argmax(power_per_freq_band)
    return int(convert_int_to_freq(max_power_ind, spec.shape[0], min_freq, max_freq))

def compute_call_interval(prediction: types.Prediction, previous: Optional[types.Prediction] = None, **_) -> float:
    if previous is None:
        return numpy.nan
    return round(prediction["start_time"] - previous["end_time"], 5)

FEATURES: Dict[str, types.FeatureExtractor] = {
    "duration": compute_duration,
    "low_freq_bb": compute_low_freq,
    "high_freq_bb": compute_high_freq,
    "bandwidth": compute_bandwidth,
    "max_power_bb": compute_max_power_bb,
    "max_power": compute_max_power,
    "max_power_first": compute_max_power_first,
    "max_power_second": compute_max_power_second,
    "call_interval": compute_call_interval}

def get_feats(spec: numpy.ndarray, pred_nms: types.PredictionResults, params: types.FeatureExtractionParameters):
    num_detections = len(pred_nms["det_probs"])
    features = numpy.empty((num_detections, len(FEATURES)), dtype=numpy.float32)
    previous = None

    for row in range(num_detections):
        prediction: types.Prediction = {
            "det_prob": float(pred_nms["det_probs"][row]),
            "class_prob": pred_nms["class_probs"][:, row],
            "start_time": float(pred_nms["start_times"][row]),
            "end_time": float(pred_nms["end_times"][row]),
            "low_freq": float(pred_nms["low_freqs"][row]),
            "high_freq": float(pred_nms["high_freqs"][row]),
            "x_pos": int(pred_nms["x_pos"][row]),
            "y_pos": int(pred_nms["y_pos"][row]),
            "bb_width": int(pred_nms["bb_width"][row]),
            "bb_height": int(pred_nms["bb_height"][row])}

        for col, feature in enumerate(FEATURES.values()):
            features[row, col] = feature(prediction, previous=previous, spec=spec, **params)
        previous = prediction
    return features
    
    
class Classifier():
    """Uses BatDetect2 lower level code without modification any modifications are in this class"""
    def __init__(self): 
        args = {'cnn_features': False, 'spec_features': False, 'quiet': False, 'save_preds_if_empty': False, 'model_path': DEFAULT_MODEL_PATH}
        code_dir = os.path.dirname(os.path.abspath(__file__))
        self.model, self.modelParams = load_model(os.path.join(code_dir, DEFAULT_MODEL_PATH)) 
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
        """Creates file species summary info"""
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
            if id == "Barbastellus barbastellus": id = "Barbastella barbastellus" #batdetect2 latine error
            if self.latinToLangDict is None: species = id
            else: species = self.latinToLangDict[id]
            if val[0] == 1: summary += f"{species} 1 call {val[1]:.0%}, "
            else: summary += f"{species} {val[0]} calls {val[2]:.0%}-{val[1]:.0%}, "
        return summary
        
    def save_results_to_file(self, results, op_path: str) -> None:
        """Creates call annotation file"""
        summary = ""
        if len(results) > 0:
            result_list = results["pred_dict"]["annotation"] # save csv file - if there are predictions
            if "SRD0" in op_path:
                # Smith Robotics Device Demeter has interference at Horseshoe frequency
                Demeter_result_list = []
                for call in result_list:
                    if call['class'] != 'Rhinolophus ferrumequinum':
                        Demeter_result_list.append(call)
                result_list = Demeter_result_list
            results_df = pandas.DataFrame(result_list)
            results_df["file_name"] = results["pred_dict"]["id"] # add file name as a column
            results_df.index.name = "id" # rename index column   
            if "class_prob" in results_df.columns:  # create a csv file with predicted events
                preds_df = results_df[["det_prob", "start_time",  "end_time", "high_freq", "low_freq", "class", "class_prob", "event"]]
                preds_df.to_csv(op_path + ".csv", sep=",")
                summary = self.GetDfSummary(preds_df)
                #create file for training as well
                with open(op_path + ".json", "w", encoding="utf-8") as jsonfile:
                    json.dump(results["pred_dict"], jsonfile, indent=2)
            else:
                with open(op_path + ".csv", "w") as f:
                    f.write("id,det_prob,start_time,end_time,high_freq,low_freq,class,class_prob\n")
        else:
            with open(op_path + ".csv", "w") as f: # empty file so do not repeat classification
                f.write("id,det_prob,start_time,end_time,high_freq,low_freq,class,class_prob\n")
        return summary
 
    def process_file(self, audio_file: str, model: DetectionModel, device: torch.device = DEVICE) -> Union[RunResults, Any]:
        """Replaces function of same name in BatDetect2"""
        predictions = []; spec_feats = []
        info = soundfile.info(audio_file)
        file_samp_rate = info.samplerate
        filename = os.path.basename(os.path.splitext(audio_file)[0])
        if filename.endswith("TE"): timeExpFact = 10
        else: timeExpFact = 1
        orig_samp_rate = file_samp_rate * timeExpFact
        sampling_rate, audio_full = load_audio(audio_file, time_exp_fact=timeExpFact,  target_samp_rate=TARGET_SAMPLERATE_HZ)
        
        for chunk_time, audio in iterate_over_chunks(audio_full, sampling_rate, CHUNK_SIZE):
            pred_nms, features, spec = _process_audio_array( audio, sampling_rate, model, self.modelParams, device)
            pred_nms["start_times"] += chunk_time
            pred_nms["end_times"] += chunk_time
            predictions.append(pred_nms)

            # extract features - if there are any calls detected
            if pred_nms["det_probs"].shape[0] == 0:
                continue

        # Merge results from chunks
        predictions = _merge_results(predictions)

        # convert results to a dictionary in the right format
        calls = convert_results(file_id=os.path.basename(audio_file), time_exp=1, # assume display also expanding
            duration=audio_full.shape[0] / float(sampling_rate), params=self.modelParams, predictions=predictions, nyquist_freq=orig_samp_rate / 2)
        return calls

    def File(self, filepath, debug=False):
        """Classifies one file using BatDetect2"""
        dir = os.path.dirname(filepath)
        file = os.path.basename(filepath)
        calls = self.process_file(filepath, self.model)
        op_dir = os.path.join(dir,"ann")
        if not os.path.isdir(op_dir): # make directory if it does not exist
            os.makedirs(op_dir)
        summary = self.save_results_to_file(calls, os.path.join(op_dir ,file)) # empty file saves trying to classify again
        if len(summary)> 0: print(colorama.Fore.GREEN + f"{file}, {summary}  " + colorama.Fore.RESET, flush=True)
        return summary