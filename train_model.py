import argparse, json, warnings, numpy, torch, datetime, os, glob, copy
import torchaudio, librosa, traceback, colorama, inspect, wakepy, random, math
import Net2dFast, Classifier

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(threshold=torch.inf, linewidth=200, precision=3)
numpy.set_printoptions(threshold=numpy.inf)

DEBUG = False
DETECTION_OVERLAP = 0.01  # has to be within this number of ms to count as detection
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 500
NUM_SAVE_EPOCHS = 50
TRAIN_FILE_USED_SEC = 1   # standarised length in seconds
SPEC_TRAIN_WIDTH = 2560   # equivalent to 1 seoond,  units are number of time steps (before resizing is performed)
TARGET_SIGMA = 2.0        #value used by draw_gaussian

DET_LOSS_WEIGHT = 1.0     # weight for the detection part of the loss
SIZE_LOSS_WEIGHT = 0.1    # weight for the bbox size loss
CLASS_LOSS_WEIGHT  = 2.0  # weight for the classification loss	

AUGMENT = True
AUG_PROB = 0.15
ECHO_MAX_DELAY = 0.005          # simulate echo by adding copy of raw audio
STRETCH_SQUEEZE_DELTA = 0.04    # stretch or squeeze spec
MASK_MAX_TIME_PERC = 0.05       # max mask size - here percentage, not ideal
MASK_MAX_FREQ_PERC = 0.10       # max mask size - here percentage, not ideal
SPEC_AMP_SCALING = 2.0 

def summarize_array(name, value):
    if not DEBUG: return
    if numpy.isscalar(value):
        print(colorama.Style.BRIGHT + f"{name + colorama.Style.RESET_ALL}: {value}")
        return
    try:
        arr = numpy.asarray(value)  # Convert lists or other sequences to NumPy array
    except Exception as e:
        print(fcolorama.Back.RED + "{name}: Could not convert to NumPy array ({e})")
        return
    flat = arr.flatten()
    nan_count = numpy.isnan(flat).sum() if numpy.issubdtype(arr.dtype, numpy.floating) else 0
    print(colorama.Style.BRIGHT + f"{name + colorama.Style.RESET_ALL}: shape={arr.shape}, {arr.dtype}, min={numpy.min(flat):.3f}, max={numpy.max(flat):.3f} mean={numpy.mean(flat):.3f}, NANs={int(nan_count)}")

def summarize(functionName, data):
    if not DEBUG: return
    if isinstance(data, numpy.ndarray) or hasattr(data, "__array__"):
        callers_locals = inspect.currentframe().f_back.f_locals
        # Find all variable names that reference the same object
        names = [name for name, val in callers_locals.items() if val is var]
        summarize_array(names[0], data)
    elif isinstance(data, dict):
        if DEBUG: print(functionName)
        for key, value in data.items():          
            summarize_array(str(key), value) 
    else: print(colorama.Back.RED + "summarize: Unsupported data type. Please provide a NumPy array or a dictionary of arrays." + colorama.Back.RESET)
    
def load_set_of_anns(wav_path):
    audioFiles = glob.glob(os.path.join(wav_path, "**", "*.wav"), recursive=True)
    anns = []
    for path in audioFiles:
        #if "Noise" in path: continue
        jsonFilepath = os.path.join(os.path.dirname(path), "ann", os.path.basename(path) + ".json")
        try:
            with open(jsonFilepath) as da:
                ann = json.load(da)
            dir = jsonFilepath[: jsonFilepath.find(os.sep + "ann" + os.sep)]
            ann["file_path"] = path
            anns.append(ann)
        except Exception as e:
            print(colorama.Back.YELLOW + colorama.Fore.BLACK + f"[WARNING] {e} for {jsonFilepath=}" + colorama.Fore.RESET + colorama.Back.RESET)
     # get unique class names
    class_names_all = []
    for ann in anns:
        for aa in ann["annotation"]:
            class_names_all.append(CompositeClass(aa["class"], aa["event"]))
    class_names, class_cnts = numpy.unique(class_names_all, return_counts=True)
    class_inv_freq = class_cnts.sum() / (len(class_names) * class_cnts.astype(numpy.float32))
    print("load_set_of_anns Class count:")
    str_len = numpy.max([len(cc) for cc in class_names]) + 5
    for cc in range(len(class_names)):
        print(str(cc).ljust(5) + class_names[cc].ljust(str_len) + str(class_cnts[cc]))
    return anns, class_names.tolist(), class_inv_freq 
    
def get_params():
    params = {}
    params["class_names"] = []
    return params

#batdetect2.train.audio_dataloader AudioLoader
def echo_aug(audio, sampling_rate):
    if DEBUG: print(f"echo_aug")
    sample_offset = ( int(ECHO_MAX_DELAY * numpy.random.random() * sampling_rate) + 1)
    audio[:-sample_offset] += numpy.random.random() * audio[sample_offset:]
    return audio
    
def mask_time_aug(spec):
    # Mask out a random block of time - repeat up to 3 times
    if DEBUG: print(f"mask_time_aug")
    fm = torchaudio.transforms.TimeMasking(int(spec.shape[1] * MASK_MAX_TIME_PERC))
    for ii in range(numpy.random.randint(1, 4)):
        spec = fm(spec)
    return spec

def mask_freq_aug(spec):
    # Mask out a random frequncy range - repeat up to 3 times
    if DEBUG: print(f"mask_freq_aug")
    fm = torchaudio.transforms.FrequencyMasking(int(spec.shape[1] * MASK_MAX_FREQ_PERC))
    for ii in range(numpy.random.randint(1, 4)):
        spec = fm(spec)
    return spec

def scale_vol_aug(spec):
    if DEBUG: print(f"scale_vol_aug")
    return spec * numpy.random.random() * SPEC_AMP_SCALING

def warp_spec_aug(spec, ann):
    # Augment spectrogram by randomly stretch and squeezing
    # NOTE this also changes the start and stop time in place not taking care of spec for viz
    if DEBUG: print(f"warp_spec_aug")
    op_size = (spec.shape[1], spec.shape[2])
    resize_fract_r = numpy.random.rand() * STRETCH_SQUEEZE_DELTA * 2 - STRETCH_SQUEEZE_DELTA + 1.0
    resize_amt = int(spec.shape[2] * resize_fract_r)
    if resize_amt >= spec.shape[2]:
        spec_r = torch.cat((spec, torch.zeros((1, spec.shape[1], resize_amt - spec.shape[2]), dtype=spec.dtype,)), 2)
    else: spec_r = spec[:, :, :resize_amt]
    spec = torch.nn.functional.interpolate(spec_r.unsqueeze(0), size=op_size, mode="bilinear", align_corners=False).squeeze(0)
    ann["start_times"] *= 1.0 / resize_fract_r
    ann["end_times"] *= 1.0 / resize_fract_r
    return spec
    
def random_bandpass_filter(spec, low_factor=0.5, high_factor=0.5):
    """spec: torch.Tensor (freq x time) or (batch x freq x time)"""
    F = spec.shape[-2]
    low = random.uniform(0, low_factor)
    high = random.uniform(1 - high_factor, 1)
    freqs = torch.linspace(0, 1, F, device=spec.device)
    center = (low + high) / 2
    width = (high - low) / 4
    response = torch.exp(-((freqs - center) ** 2) / (2 * width ** 2))
    # Expand for batch if needed
    if spec.dim() == 3:
        response = response.unsqueeze(0)
    return spec * response.unsqueeze(-1)    

def random_time_shift(spec, max_shift_frames=20):
    """S: torch.Tensor (freq x time) or (batch x freq x time)"""
    shift = random.randint(-max_shift_frames, max_shift_frames)
    if shift == 0:
        return spec
    if shift > 0:
        pad = (shift, 0)
        spec_pad = torch.nn.functional.pad(spec, pad, mode='constant', value=0)
        return spec_pad[..., :-shift]
    else:
        pad = (0, -shift)
        spec_pad = torch.nn.functional.pad(spec, pad, mode='constant', value=0)
        return spec_pad[..., -shift:]

def dynamic_scale_factor(w):

    radius = int(base_radius * scale)
    sigma = max(1.0, radius / 3.0)
    sigma = min(sigma, 12.0)
    return sigma
        
def gaussian_sigma_from_box(h, w, min_overlap=0.7, scale=1.5,  max_radius=80,  max_sigma=12.0):
    """ Computes a CornerNet-style Gaussian radius from bounding box size,
    applies scaling for large calls, converts to sigma, and caps it.
    Returns sigma for draw_gaussian(). """
    # --- CornerNet radius calculation ---
    if w < 80: scale = 1.5
    elif w < 200: scale = 2.0
    elif w < 400: scale = 3.0
    else: scale = 4.0
    a1 = 1
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0, b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = math.sqrt(max(0, b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = math.sqrt(max(0, b3 ** 2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2
    base_radius = min(r1, r2, r3)
    # --- Scale for large calls (social calls, feeding buzzes) ---
    radius = int(base_radius * scale)
    radius = min(radius, max_radius)
    # --- Convert radius → sigma ---
    sigma = radius / 3.0
    # --- Cap sigma to avoid flattening ---
    sigma = max(1.0, min(sigma, max_sigma))
    return sigma
    
def draw_gaussian(heatmap, center, sigmax, sigmay=None):
    """CornerNet is a novel approach to object detection that detects objects as pairs of corners top-left and bottom-right) using a single convolutional neural network.
    It eliminates the need for predefined anchor boxes, which simplifies the detection process.
    The network predicts heatmaps for these corners, and focal loss is used to optimize the prediction of these heatmaps. 
    CornerNet achieves high accuracy by associating detected corners with similar embeddings, enhancing the detection of objects in images"""
    if sigmay is None:
        sigmay = sigmax
    tmp_size = numpy.maximum(sigmax, sigmay) * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return False
    size = 2 * tmp_size + 1
    x = numpy.arange(0, size, 1, numpy.float32)
    y = x[:, numpy.newaxis]
    x0 = y0 = size // 2
    g = numpy.exp(-((x - x0) ** 2) / (2 * sigmax**2) - ((y - y0) ** 2) / (2 * sigmay**2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = numpy.maximum(heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],  g[g_y[0] : g_y[1], g_x[0] : g_x[1]])
    return True
    
def time_to_x_coords(time_in_file: float, samplerate: float, window_duration: float, window_overlap: float) -> float:
    nfft = numpy.floor(window_duration * samplerate)  # int() uses floor
    noverlap = numpy.floor(window_overlap * nfft)
    return (time_in_file * samplerate - noverlap) / (nfft - noverlap)

def target_heatmaps(spec_op_shape: Tuple[int, int], sampling_rate: int, ann: AnnotationGroup, class_names) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, AnnotationGroup]:
    # spec may be resized on input into the network
    num_classes = len(class_names)
    op_height = spec_op_shape[0]
    op_width = spec_op_shape[1]
    freq_per_bin = (Classifier.MAX_FREQ_HZ - Classifier.MIN_FREQ_HZ) / op_height
    # start and end times
    x_pos_start = time_to_x_coords(ann["start_times"], sampling_rate, Classifier.FFT_WIN_LENGTH_S, Classifier.FFT_OVERLAP)
    x_pos_start = (Classifier.RESIZE_FACTOR * x_pos_start).astype(numpy.int32)
    x_pos_end = time_to_x_coords(ann["end_times"], sampling_rate, Classifier.FFT_WIN_LENGTH_S, Classifier.FFT_OVERLAP)
    x_pos_end = (Classifier.RESIZE_FACTOR * x_pos_end).astype(numpy.int32)
    # location on y axis i.e. frequency
    y_pos_low = (ann["low_freqs"] - Classifier.MIN_FREQ_HZ) / freq_per_bin
    y_pos_low = (op_height - y_pos_low).astype(numpy.int32)
    y_pos_high = (ann["high_freqs"] - Classifier.MIN_FREQ_HZ) / freq_per_bin
    y_pos_high = (op_height - y_pos_high).astype(numpy.int32)
    bb_widths = x_pos_end - x_pos_start
    bb_heights = y_pos_low - y_pos_high
    # Only include annotations that are within the input spectrogram
    valid_inds = numpy.where((x_pos_start >= 0) & (x_pos_start < op_width) & (y_pos_low >= 0)  & (y_pos_low < (op_height - 1)))[0]
    ann_aug: AnnotationGroup = {
        "start_times": ann["start_times"][valid_inds],
        "end_times": ann["end_times"][valid_inds],
        "high_freqs": ann["high_freqs"][valid_inds],
        "low_freqs": ann["low_freqs"][valid_inds],
        "class_ids": ann["class_ids"][valid_inds],
        "individual_ids": ann["individual_ids"][valid_inds],
    }
    ann_aug["x_inds"] = x_pos_start[valid_inds]
    ann_aug["y_inds"] = y_pos_low[valid_inds]
    before = len(ann["start_times"]); after = len(ann_aug["start_times"])
    #if  before > after:
    # if the number of calls is only 1, then it is unique
    # TODO would be better if we found these unique calls at the merging stage
    if len(ann_aug["individual_ids"]) == 1:
        ann_aug["individual_ids"][0] = 0
    y_2d_det = numpy.zeros((1, op_height, op_width), dtype=numpy.float32)
    y_2d_size = numpy.zeros((2, op_height, op_width), dtype=numpy.float32)
    
    # num classes and "background" class
    y_2d_classes: numpy.ndarray = numpy.zeros((num_classes + 1, op_height, op_width), dtype=numpy.float32)
    # create 2D ground truth heatmaps
    for ii in valid_inds:
        heatmap = y_2d_det[0, :]
        w, h = heatmap.shape[0], heatmap.shape[1]
        #radius = gaussian_radius_fixed(h, w)
        #sigma = max(1.0, radius / 3.0)
        sigma = gaussian_sigma_from_box(h, w)
        draw_gaussian(y_2d_det[0, :], (x_pos_start[ii], y_pos_low[ii]), sigma)
        #draw_gaussian(y_2d_det[0, :], (x_pos_start[ii], y_pos_low[ii]), TARGET_SIGMA)
        y_2d_size[0, y_pos_low[ii], x_pos_start[ii]] = bb_widths[ii]
        y_2d_size[1, y_pos_low[ii], x_pos_start[ii]] = bb_heights[ii]
        cls_id = ann["class_ids"][ii]
        if DEBUG: print(f"target_heatmaps ii={int(ii)} {cls_id=} {class_names[int(cls_id)]}")
        if cls_id > -1:
            heatmap = y_2d_classes[cls_id, :]
            w, h = heatmap.shape[0], heatmap.shape[1]
            sigma = gaussian_sigma_from_box(h, w)
            draw_gaussian(y_2d_classes[cls_id, :], (x_pos_start[ii], y_pos_low[ii]), sigma)
            #drawResult = draw_gaussian(y_2d_classes[cls_id, :], (x_pos_start[ii], y_pos_low[ii]), TARGET_SIGMA)
            y_2d_classes_cls_id = y_2d_classes[cls_id, :]
    # be careful as this will have a 1.0 places where we have event but dont know gt class this will be masked in training anyway
    y_2d_classes[num_classes, :] = 1.0 - y_2d_classes.sum(0)
    y_2d_classes = y_2d_classes / y_2d_classes.sum(0)[numpy.newaxis, ...]
    y_2d_classes[numpy.isnan(y_2d_classes)] = 0.0
    return y_2d_det, y_2d_size, y_2d_classes, ann_aug

def resample_audio(num_samples, sampling_rate, audio2, sampling_rate2):
    if sampling_rate != sampling_rate2:
        if DEBUG: print(f"resample_audio {sampling_rate=} {sampling_rate2=}")
        audio2 = librosa.resample(audio2,  orig_sr=sampling_rate2, target_sr=sampling_rate, res_type="polyphase")
        sampling_rate2 = sampling_rate
    if audio2.shape[0] < num_samples:
        audio2 = numpy.hstack((audio2,  numpy.zeros((num_samples - audio2.shape[0]), dtype=audio2.dtype)))
    elif audio2.shape[0] > num_samples:
        audio2 = audio2[:num_samples]
    return audio2, sampling_rate2
    
def combine_audio_aug(audio, sampling_rate, ann, audio2, sampling_rate2, ann2):
    # resample so they are the same
    audio2, sampling_rate2 = resample_audio(audio.shape[0], sampling_rate, audio2, sampling_rate2)

    if (ann["annotated"] and (ann2["annotated"]) and (sampling_rate2 == sampling_rate) and (audio.shape[0] == audio2.shape[0])):
        comb_weight = 0.3 + numpy.random.random() * 0.4
        audio = comb_weight * audio + (1 - comb_weight) * audio2
        inds = numpy.argsort(numpy.hstack((ann["start_times"], ann2["start_times"])))
        for kk in ann.keys():
            # when combining calls from different files, assume they come from different individuals
            if kk == "individual_ids":
                if (ann[kk] > -1).sum() > 0:
                    ann2[kk][ann2[kk] > -1] += numpy.max(ann[kk][ann[kk] > -1]) + 1
            if (kk != "class_id_file") and (kk != "annotated"):
                ann[kk] = numpy.hstack((ann[kk], ann2[kk]))[inds]
    return audio, ann

def CompositeClass(species, call_type="Echolocation"):
    if species == "Barbastellus barbastellus": species = "Barbastella barbastellus" #error in BatDetect2 data
    return species + "-" + call_type
    
class AudioLoader(torch.utils.data.Dataset):
    def __init__(self, data_anns_ip, params, dataset_name=None, is_train=False):
        self.data_anns = []
        self.audio_file = []
        self.is_train = is_train
        self.params = params
        for ii in range(len(data_anns_ip)):
            dd = copy.deepcopy(data_anns_ip[ii])
            # filter out unused annotation here
            filtered_annotations = []
            for ii, aa in enumerate(dd["annotation"]):  
                if "individual" in aa.keys():
                    aa["individual"] = int(aa["individual"])
                    # if only one call labeled it has to be from the same individual
                    if len(dd["annotation"]) == 1:
                        aa["individual"] = 0
                # convert class name into class label
                compositeClass = CompositeClass(aa["class"], aa["event"])
                if compositeClass in self.params["class_names"]:
                    aa["class_id"] = self.params["class_names"].index(compositeClass)
                else:
                    print(colorama.Back.RED + f"AudioLoader __init__ class {compositeClass} NOT FOUND for {dd["file_path"]}" + colorama.Back.RESET)
                    aa["class_id"] = -1
                filtered_annotations.append(aa)
            dd["annotation"] = filtered_annotations
            dd["start_times"] = numpy.array([aa["start_time"] for aa in dd["annotation"]]).astype(numpy.float64)
            dd["end_times"] = numpy.array([aa["end_time"] for aa in dd["annotation"]]).astype(numpy.float64)
            dd["high_freqs"] = numpy.array([float(aa["high_freq"]) for aa in dd["annotation"]]).astype(numpy.float64)
            dd["low_freqs"] = numpy.array([float(aa["low_freq"]) for aa in dd["annotation"]]).astype(numpy.float64)
            dd["class_ids"] = numpy.array([aa["class_id"] for aa in dd["annotation"]]).astype(numpy.int32)
            dd["individual_ids"] = numpy.array([aa["individual"] for aa in dd["annotation"]]).astype(numpy.int32)
            # file level class name
            if "class_name" in dd.keys(): # file level value, call one is 'class'
                compositeClass = CompositeClass(dd["class_name"])
                if compositeClass in self.params["class_names"]:
                    dd["class_id_file"] = self.params["class_names"].index(compositeClass)
                else: 
                    print(colorama.Back.RED + f"AudioLoader __init__ class_name {compositeClass} NOT FOUND for {dd["file_path"]}" + colorama.Back.RESET)
                    dd["class_id_file"] = -1
            self.data_anns.append(dd)
            self.audio_file.append(dd["file_path"])
        ann_cnt = [len(aa["annotation"]) for aa in self.data_anns]
        self.max_num_anns = 2 * numpy.max(ann_cnt)  # x2 because we may be combining files during training
        print("\n")
        if dataset_name is not None:
            print("Dataset     : " + dataset_name)
        print("Num files   : " + str(len(self.data_anns)))
        print("Num calls   : " + str(numpy.sum(ann_cnt)))

    def get_file_and_anns(self, index=None):
        # if no file specified, choose random one
        if index == None:
            index = numpy.random.randint(0, len(self.data_anns))
        audio_file = self.audio_file[index]
        sampling_rate, audio_raw = Classifier.load_audio(audio_file, self.data_anns[index]["time_exp"], Classifier.TARGET_SAMPLERATE_HZ)
        # copy annotation
        ann = {}
        ann["annotated"] = self.data_anns[index]["annotated"]
        ann["class_id_file"] = self.data_anns[index]["class_id_file"]
        keys = ["start_times", "end_times", "high_freqs", "low_freqs", "class_ids", "individual_ids"]
        for kk in keys:
            ann[kk] = self.data_anns[index][kk].copy()
        # if train then grab a random crop
        nfft = Classifier.FFT_WIN_LENGTH_S * sampling_rate
        noverlap = Classifier.FFT_OVERLAP * nfft
        target_samples = int(Classifier.TARGET_SAMPLERATE_HZ * TRAIN_FILE_USED_SEC)
        spec_train_width = int(target_samples / (nfft - noverlap) - noverlap)
        if audio_raw.shape[0] > target_samples:
            if DEBUG: print(colorama.Fore.YELLOW + f"get_file_and_anns cropping {os.path.basename(audio_file)} as {audio_raw.shape[0]=} > {target_samples=}" + colorama.Fore.RESET)
            sample_crop = numpy.random.randint(audio_raw.shape[0] - target_samples)
            audio_raw = audio_raw[sample_crop : sample_crop + target_samples]
            ann["start_times"] = ann["start_times"] - sample_crop / float(sampling_rate)
            ann["end_times"] = ann["end_times"] - sample_crop / float(sampling_rate)
        # pad audio
        op_spec_target_size = spec_train_width
        audio_raw = Classifier.pad_audio(audio_raw, sampling_rate, Classifier. FFT_WIN_LENGTH_S, Classifier.FFT_OVERLAP, Classifier.RESIZE_FACTOR, Classifier.SPEC_DIVIDE_FACTOR, op_spec_target_size)
        duration = audio_raw.shape[0] / float(sampling_rate)
        # sort based on time
        inds = numpy.argsort(ann["start_times"])
        for kk in ann.keys():
            if (kk != "class_id_file") and (kk != "annotated"):
                ann[kk] = ann[kk][inds]
        summarize("get_file_and_anns " + os.path.basename(audio_file), ann)
        return audio_raw, sampling_rate, duration, ann
    
    def __getitem__(self, index):
        try:
            # load audio file
            if DEBUG: print()
            audio, sampling_rate, duration, ann = self.get_file_and_anns(index)
            if AUGMENT:
                # augment on raw audio - combine with random audio file
                if numpy.random.random() < AUG_PROB:
                    (audio2, sampling_rate2, duration2, ann2) = self.get_file_and_anns()
                    audio, ann = combine_audio_aug(audio, sampling_rate, ann, audio2, sampling_rate2, ann2)
                # simulate echo by adding delayed copy of the file
                if numpy.random.random() < AUG_PROB:
                    audio = echo_aug(audio, sampling_rate)
            # create spectrogram
            spec = Classifier.generate_spectrogram(audio, sampling_rate)
            spec_op_shape = (int(Classifier.SPEC_HEIGHT * Classifier.RESIZE_FACTOR), int(spec.shape[1] * Classifier.RESIZE_FACTOR))
            # resize the spec
            spec = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            spec = torch.nn.functional.interpolate(spec, size=spec_op_shape, mode="bilinear", align_corners=False).squeeze(0)
            # augment spectrogram
            if AUGMENT:
                if numpy.random.random() < AUG_PROB: spec = scale_vol_aug(spec)
                #if numpy.random.random() < AUG_PROB: spec = warp_spec_aug(spec, ann) # causes horseshoe bat problems
                if numpy.random.random() < AUG_PROB: spec = mask_time_aug(spec)
                #if numpy.random.random() < AUG_PROB: spec = mask_freq_aug(spec) # causes horseshoe bat problems
                if numpy.random.random() < AUG_PROB: spec = random_bandpass_filter(spec)
                if numpy.random.random() < AUG_PROB: spec = random_time_shift(spec)
            outputs = {}
            outputs["spec"] = spec
            # create ground truth heatmaps
            (outputs["y_2d_det"],  outputs["y_2d_size"], outputs["y_2d_classes"], ann_aug) = target_heatmaps(spec_op_shape, sampling_rate, ann, self.params["class_names"])
            # hack to get around requirement that all vectors are the same length in the output batch
            pad_size = self.max_num_anns - len(ann_aug["individual_ids"])
            outputs["is_valid"] = numpy.hstack((numpy.ones(len(ann_aug["individual_ids"])), numpy.ones(pad_size, dtype=numpy.int32) * -1))
            keys = ["class_ids", "individual_ids", "x_inds", "y_inds", "start_times",  "end_times", "low_freqs", "high_freqs"]
            for kk in keys:
                outputs[kk] = numpy.hstack((ann_aug[kk], numpy.ones(pad_size, dtype=numpy.int32) * -1))
            # convert to pytorch
            for kk in outputs.keys():
                if type(outputs[kk]) != torch.Tensor:
                    outputs[kk] = torch.from_numpy(outputs[kk])
            # scalars
            outputs["class_id_file"] = ann["class_id_file"]
            outputs["annotated"] = ann["annotated"]
            outputs["duration"] = duration
            outputs["sampling_rate"] = sampling_rate
            outputs["file_id"] = index
            return outputs
        except Exception as e:
            print(colorama.Back.RED + f"[ERROR] Failed to load {index=}" + colorama.Back.RESET)
            traceback.print_exc()
            raise
    def __len__(self):
        return len(self.data_anns)

def focal_loss_per_class(p_class, target_class, valid_mask, gamma=2.0, eps=1e-5):
    """ p_class:        (B, C, A, T)  focal loss per element
    target_class: (B, C, A, T)  one-hot targets
    valid_mask:   (B, 1, A, T)  mask of valid positions
    returns:      (C,)          one scalar loss per class """
    p = torch.clamp(p_class, eps, 1 - eps)
    ce = -target_class * torch.log(p)
    focal = (1 - p) ** gamma * ce          # (B, C, A, T)
    focal = focal * valid_mask             # mask invalid
    # total positives (or normaliser)
    num_pos = (target_class * valid_mask).sum() + eps
    # original scalar loss (what you see as ~10.496)
    loss_scalar = focal.sum() / num_pos
    
    # sum over batch, anchors, time → keep class
    per_class_sum = focal.sum(dim=(0, 2, 3))          # (C,)
    # count positives per class
    class_counts = (target_class * valid_mask).sum(dim=(0, 2, 3)) + eps  # (C,)
    # per-class *average* loss
    per_class_avg = per_class_sum / class_counts      # (C,)

    per_class_contrib = per_class_sum / num_pos
    print(f"focal_loss_per_class {loss_scalar.item()=} {per_class_avg=} {per_class_contrib=} {per_class_contrib.sum()=}")
    return

def focal_loss(pred, gt, valid_mask=None, IsClass=False):
    """ Focal loss adapted from CornerNet: Detecting Objects as Paired Keypoints
    pred  (batch x c x h x w)
    gt    (batch x c x h x w)"""
    if DEBUG: print(f"focal_loss {pred.shape=} {gt.shape=}")
    eps = 1e-5
    beta = 4
    alpha = 2 # Balances the importance of positive and negative samples.
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = (torch.log(1 - pred + eps) * torch.pow(pred, alpha) * torch.pow(1 - gt, beta) * neg_inds)
    if valid_mask is not None:
        pos_loss = pos_loss * valid_mask
        neg_loss = neg_loss * valid_mask
    pos_loss_sum = pos_loss.sum()
    neg_loss_sum = neg_loss.sum()
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = -neg_loss_sum
    else:
        loss = -(pos_loss_sum + neg_loss_sum) / num_pos
    if IsClass:
        per_class_pos_sum = pos_loss.float().sum(dim=(0, 2, 3))
        per_class_neg_sum = neg_loss.float().sum(dim=(0, 2, 3))
        #num_pos = pos_inds.float().sum(dim=(0, 2, 3)) #####
        class_counts = (gt * valid_mask).sum(dim=(0, 2, 3)) + eps
        if num_pos == 0:
            ##### Boolean value of Tensor with more than one value is ambiguous #####
            per_class_loss = -per_class_neg_sum
        else:
            per_class_loss = -(per_class_pos_sum + per_class_neg_sum) / num_pos
        #print(f"focal_loss loss={loss.item()} num_pos={num_pos.item()} per_class_loss.sum={per_class_loss.sum().item()}")
        #return loss, per_class_loss
        return per_class_loss
    return loss

def bbox_size_loss(pred_size, target_size):
    """Bounding box size loss. Only compute loss where there is a bounding box."""
    target_size_mask = (target_size > 0).float()
    return torch.nn.functional.l1_loss(pred_size * target_size_mask, target_size, reduction="sum") / (target_size_mask.sum() + 1e-5)

def loss_fun(outputs, target_det, target_size, target_class, class_inv_freq):
    detectionLoss = DET_LOSS_WEIGHT * focal_loss(outputs.pred_det, target_det)  
    boundingBoxSizeLoss = SIZE_LOSS_WEIGHT * bbox_size_loss(outputs.pred_size, target_size)
    summarize_array("loss_fun", target_class) 
    valid_mask = (target_class[:, :-1, :, :].sum(1) > 0).float().unsqueeze(1) 
    summarize_array("loss_fun", valid_mask)
    p_class = outputs.pred_class[:, :-1, :]
    per_class_loss = focal_loss(p_class, target_class[:, :-1, :], valid_mask=valid_mask, IsClass=True)
    #class_loss, per_class_loss = focal_loss(p_class, target_class[:, :-1, :], valid_mask=valid_mask, IsClass=True)
    #class_loss = CLASS_LOSS_WEIGHT * class_loss; 
    per_class_loss = per_class_loss * CLASS_LOSS_WEIGHT
    return detectionLoss, boundingBoxSizeLoss, per_class_loss
    #return detectionLoss, boundingBoxSizeLoss, class_loss, per_class_loss

def train(model, epoch, data_loader, optimizer, scheduler, params):
    model.train()
    
    class_inv_freq = torch.from_numpy(numpy.array(params["class_inv_freq"], dtype=numpy.float32)).to(params["device"])
    class_inv_freq = class_inv_freq.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    det_loss_sum = size_loss_sum = 0; 
    #class_loss_sum = 0
    per_class_loss_sum = torch.zeros(len(params["class_names"])).to(params["device"])
    count = 0
    for batch_idx, inputs in enumerate(data_loader):
        try:
            data = inputs["spec"].to(params["device"])
            target_det = inputs["y_2d_det"].to(params["device"])
            target_size = inputs["y_2d_size"].to(params["device"])
            target_class = inputs["y_2d_classes"].to(params["device"])
            optimizer.zero_grad()
            outputs = model(data)
            det_loss, size_loss, per_class_loss = loss_fun(outputs, target_det, target_size, target_class, class_inv_freq)
            #det_loss, size_loss, class_loss, per_class_loss = loss_fun(outputs, target_det, target_size, target_class, class_inv_freq)
            det_loss_sum += det_loss.item() * data.shape[0]; size_loss_sum += size_loss.item() * data.shape[0] 
            #; class_loss_sum += class_loss.item() * data.shape[0]
            per_class_loss_sum = per_class_loss_sum + (per_class_loss * data.shape[0])
            #print(f"train {class_loss_sum=} {data.shape[0]=} {per_class_loss_sum=} {per_class_loss=}")
            count += data.shape[0]
            loss = det_loss + size_loss + per_class_loss.sum()  ############# what matters ################
            loss.backward()
            optimizer.step()
            scheduler.step()
        except Exception as e:
            print(colorama.Back.BLUE + f"[WARNING] Skipping batch {batch_idx}: {e}" + colorama.Back.RESET)
            traceback.print_exc()
            continue
    #print(f"train class_loss_sum={class_loss_sum} per_class_loss_sum.sum={per_class_loss_sum.sum().item()} {per_class_loss_sum.shape=}")
    if epoch % NUM_SAVE_EPOCHS == 0:
        classes = params["class_names"]
        for idx, value in enumerate(per_class_loss_sum):
            print(f"{epoch=} Train {classes[idx]} {value:.3f}")

    det_loss_avg = det_loss_sum / count; size_loss_avg = size_loss_sum / count; class_loss_avg = per_class_loss_sum.sum() / count
    #det_loss_avg = det_loss_sum / count; size_loss_avg = size_loss_sum / count; class_loss_avg = class_loss_sum / count
    train_loss = det_loss_avg + size_loss_avg + class_loss_avg
    print(f"{epoch=} Train loss {train_loss:.3f} = detection {det_loss_avg:.3f} + box size {size_loss_avg:.3f} + class {class_loss_avg:.3f}")
    return float(train_loss)
    #res = {}
    #res["train_loss"] = float(train_loss)
    #return res

def main():
    params = get_params()
    if torch.cuda.is_available(): params["device"] = "cuda"
    else: params["device"] = "cpu"

    # setup arg parser and populate it with exiting parameters - will not work with lists
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to root of datasets")
    for key, val in params.items():
        parser.add_argument("--" + key, type=type(val), default=val)
    params = vars(parser.parse_args())
    if torch.cuda.is_available(): print(colorama.Fore.GREEN + "torch.cuda.is_available" + colorama.Fore.RESET)
    else: print(colorama.Fore.RED + "torch.cuda is not available" + colorama.Fore.RESET)
    
    with wakepy.keep.running():
        min_loss = 1.79e308
        (data_train, params["class_names"], class_inv_freq) = load_set_of_anns(params["data_dir"])
        params["class_inv_freq"] = class_inv_freq.tolist()
        # train loader
        train_dataset = AudioLoader(data_train, params, is_train=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,)
        inputs_train = next(iter(train_loader))
        params["ip_height"] = int(Classifier.SPEC_HEIGHT * Classifier.RESIZE_FACTOR)
        print("\ntrain batch spec size :", inputs_train["spec"].shape)
        print("class target size     :", inputs_train["y_2d_classes"].shape)
        # select network
        num_classes = len(params["class_names"])
        model = Net2dFast.Net2dFast(Classifier.NUM_FILTERS,num_classes=num_classes, ip_height=params["ip_height"])
        model = model.to(Classifier.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS * len(train_loader))
        # save parameters to file 
        with open(os.path.join(params["data_dir"], "params.json"), "w") as da:
            json.dump(params, da, indent=2, sort_keys=True)       
        # main train loop
        for epoch in range(0, NUM_EPOCHS + 1):
            train_loss = train(model, epoch, train_loader, optimizer,  scheduler, params)
            if train_loss < min_loss:
                best_model = model
                min_loss = train_loss
            if epoch % NUM_SAVE_EPOCHS == 0:
                # save trained model
                now_str = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                model_file_name = f"E{epoch}_AUG_{now_str}.pth.tar"
                print(f"saving model to: {model_file_name}")
                op_state = {"epoch": epoch + 1, "state_dict": best_model.state_dict(), "params": params}
                torch.save(op_state, os.path.join(params["data_dir"], model_file_name))

if __name__ == "__main__":
    main()
