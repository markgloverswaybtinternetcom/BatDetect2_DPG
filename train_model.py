import argparse, json, warnings, numpy, torch, datetime, os, glob, copy, polars
import torchaudio, librosa, traceback, colorama, inspect, wakepy, random, math, scipy
import Net2dFast, Classifier

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(threshold=torch.inf, linewidth=200, precision=3)
numpy.set_printoptions(threshold=numpy.inf)

DEBUG = False
DETECTION_OVERLAP = 0.01  # has to be within this number of ms to count as detection
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_WORKERS = 4
MIN_EPOCHS = 300
MAX_EPOCHS = 900
NUM_SAVE_EPOCHS = 50
TRAIN_FILE_USED_SEC = 1   # standarised length in seconds
SPEC_TRAIN_WIDTH = 2560   # equivalent to 1 seoond,  units are number of time steps (before resizing is performed)

DET_LOSS_WEIGHT = 1.0     # weight for the detection part of the loss
SIZE_LOSS_WEIGHT = 0.1    # weight for the bbox size loss
#CLASS_LOSS_WEIGHT  = 2.0  # weight for the classification loss	
GAUSSIAN_SIGMA = 12
#Only used on first run until class difficulty found
DEFAULT_CLASS_WEIGHTS = { 
    "Barbastella barbastellus-Echolocation": 4.5, 
    "Barbastella barbastellus-Feeding Buzz": 0.0,
    "Barbastella barbastellus-Social": 3.0,
    "Eptesicus serotinus-Echolocation": 4.5, 
    "Eptesicus serotinus-Feeding Buzz": 0.0,
    "Eptesicus serotinus-Social": 3.5,
    "Myotis alcathoe-Echolocation": 2.0,
    "Myotis alcathoe-Feeding Buzz": 0.0,
    "Myotis alcathoe-Social": 2.0,
    "Myotis bechsteinii-Echolocation": 3.0,
    "Myotis bechsteinii-Social": 2.0,
    "Myotis brandtii-Echolocation": 2.5,
    "Myotis brandtii-Feeding Buzz": 0.0,
    "Myotis brandtii-Social": 2.0,
    "Myotis daubentonii-Echolocation": 2.5,
    "Myotis daubentonii-Feeding Buzz": 0.0,
    "Myotis daubentonii-Social": 3.0,
    "Myotis mystacinus-Echolocation": 3.5,
    "Myotis mystacinus-Social": 2.0,
    "Myotis nattereri-Echolocation": 3.5,
    "Myotis nattereri-Social": 3.0,
    "Nyctalus leisleri-Echolocation": 3.5,
    "Nyctalus leisleri-Social": 3.5,
    "Nyctalus noctula-Echolocation": 2.5,
    "Nyctalus noctula-Feeding Buzz": 0.0,
    "Nyctalus noctula-Social": 2.5,
    "Pipistrellus nathusii-Echolocation": 2.0,
    "Pipistrellus nathusii-Feeding Buzz": 0.0,
    "Pipistrellus nathusii-Social": 2.0,
    "Pipistrellus pipistrellus-Echolocation": 1.0,
    "Pipistrellus pipistrellus-Feeding Buzz": 0.0,
    "Pipistrellus pipistrellus-Social": 3.5,  
    "Pipistrellus pygmaeus-Echolocation": 1.0,
    "Pipistrellus pygmaeus-Feeding Buzz": 0.0,
    "Pipistrellus pygmaeus-Social": 3.5, 
    "Plecotus auritus-Echolocation": 3.5,
    "Plecotus auritus-Social": 3.5, 
    "Plecotus austriacus-Echolocation": 2.0,
    "Plecotus austriacus-Social": 2.0,
    "Rhinolophus ferrumequinum-Echolocation": 1.5,
    "Rhinolophus ferrumequinum-Social": 1.5,
    "Rhinolophus hipposideros-Echolocation": 1.5, 
    "Rhinolophus hipposideros-Social": 1.5 
}
    
AUGMENT = True
AUG_PROB = 0.15
COMBINE_PROB = 0.08 ## try 1
ECHO_MAX_DELAY = 0.005          # simulate echo by adding copy of raw audio
STRETCH_SQUEEZE_DELTA = 0.04    # stretch or squeeze spec
MASK_MAX_TIME_PERC = 0.05       # max mask size - here percentage, not ideal
MASK_MAX_FREQ_PERC = 0.10       # max mask size - here percentage, not ideal
SPEC_AMP_SCALING = 2.0 
HORSESHOE_CF = {
    "Rhinolophus ferrumequinum-Echolocation": 80000,   # Greater Horseshoe CF (Hz)
    "Rhinolophus hipposideros-Echolocation": 110000,   # Lesser Horseshoe CF (Hz)
}

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
    vals_greater_mean = (flat > numpy.mean(flat)).sum()
    print(colorama.Style.BRIGHT + f"{name + colorama.Style.RESET_ALL}: shape={arr.shape}, {arr.dtype}, min={numpy.min(flat):.3f}, max={numpy.max(flat):.3f} mean={numpy.mean(flat):.3f}, N above mean= {vals_greater_mean}, NANs={int(nan_count)}")

def next_model_number(models_dir):
    files = glob.glob(os.path.join(models_dir, "model_*.pth.tar"))
    if not files:
        return 1
    nums = []
    for f in files:
        base = os.path.basename(f)
        # model_<n>_E<epoch>.pth.tar
        try:
            n = int(base.split("_")[1])
            nums.append(n)
        except:
            continue
    return max(nums) + 1 if nums else 1
    
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
        jsonFilepath = os.path.join(os.path.dirname(path), "ann", os.path.basename(path) + ".json")
        try:
            with open(jsonFilepath) as da:
                ann = json.load(da)
            dir = jsonFilepath[: jsonFilepath.find(os.sep + "ann" + os.sep)]
            ann["file_path"] = path
            anns.append(ann)
        except Exception as e:
            print(colorama.Back.YELLOW + colorama.Fore.BLACK + f"[WARNING] {e}" + colorama.Fore.RESET + colorama.Back.RESET)
     # get unique class names
    class_names_all = []
    for ann in anns:
        for aa in ann["annotation"]:
            class_names_all.append(CompositeClass(aa["class"], aa["event"]))
    class_names, class_cnts = numpy.unique(class_names_all, return_counts=True)
    #class_inv_freq = class_cnts.sum() / (len(class_names) * class_cnts.astype(numpy.float32))
    #class_inv_freq = (total_calls / num_classes) * (1 / class_cnts)
    class_inv_freq = 1.0 / (class_cnts.astype(numpy.float32) + 1e-8)
    print("load_set_of_anns Class count:")
    str_len = numpy.max([len(cc) for cc in class_names]) + 5
    for cc in range(len(class_names)):
        print(f"{str(cc).ljust(5)}, {class_names[cc].ljust(str_len)}, {str(class_cnts[cc])}")
    return anns, class_names.tolist(), class_inv_freq 

#batdetect2.train.audio_dataloader AudioLoader
def echo_aug(audio, sampling_rate):
    sample_offset = ( int(ECHO_MAX_DELAY * numpy.random.random() * sampling_rate) + 1)
    audio[:-sample_offset] += numpy.random.random() * audio[sample_offset:]
    return audio
    
def mask_time_aug(spec):
    # Mask out a random block of time - repeat up to 3 times
    fm = torchaudio.transforms.TimeMasking(int(spec.shape[1] * MASK_MAX_TIME_PERC))
    for ii in range(numpy.random.randint(1, 4)):
        spec = fm(spec)
    return spec

def mask_freq_aug(spec):
    # Mask out a random frequncy range - repeat up to 3 times
    fm = torchaudio.transforms.FrequencyMasking(int(spec.shape[1] * MASK_MAX_FREQ_PERC))
    for ii in range(numpy.random.randint(1, 4)):
        spec = fm(spec)
    return spec
    
def scale_vol_aug(spec):
    return spec * numpy.random.random() * SPEC_AMP_SCALING

def warp_spec_aug(spec, ann):
    # Randomly stretch or squeeze the time axis only
    # Original output size (C, F, T)
    op_size = (spec.shape[1], spec.shape[2])
    # Random stretch/squeeze factor
    resize_fract_r = (numpy.random.rand() * STRETCH_SQUEEZE_DELTA * 2 - STRETCH_SQUEEZE_DELTA + 1.0)
    # New time dimension
    resize_amt = int(spec.shape[2] * resize_fract_r)
    # Pad or crop along time axis
    if resize_amt >= spec.shape[2]:
        pad = resize_amt - spec.shape[2]
        spec_r = torch.cat((spec, torch.zeros((1, spec.shape[1], pad), dtype=spec.dtype)), dim=2)
    else:
        spec_r = spec[:, :, :resize_amt]
    # Resize back to original time dimension
    spec_warped = torch.nn.functional.interpolate(spec_r.unsqueeze(0), size=op_size, mode="bilinear", align_corners=False).squeeze(0)

    # Adjust annotation times
    ann["start_times"] *= 1.0 / resize_fract_r
    ann["end_times"] *= 1.0 / resize_fract_r
    return spec_warped
    
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

def inject_vertical_noise_streak(spec, strength=0.02):
    # Choose a random frequency bin (second axis)
    freq_bin = numpy.random.randint(0, spec.shape[1])
    # Add noise across all time bins (first axis)
    spec[:, freq_bin] += strength * numpy.random.rand(spec.shape[0])
    return spec

def reinforce_cf_band(spec, cf_freq, boost_db=1.5):
    # spec.shape == (time_bins, freq_bins)
    op_height = spec.shape[1]
    freq_per_bin = (Classifier.MAX_FREQ_HZ - Classifier.MIN_FREQ_HZ) / op_height
    cf_bin = int((cf_freq - Classifier.MIN_FREQ_HZ) / freq_per_bin)
    # Safety clamp
    cf_bin = max(0, min(cf_bin, op_height - 1))
    # Boost CF band across all time bins
    spec[:, cf_bin] *= 10 ** (boost_db / 20.0)
    return spec
    
"""import matplotlib.pyplot
def debug_heatmap_alignment_corners(spec, heatmap, x1, x2, y1, y2, ii, class_name):
    heatmap = torch.from_numpy(heatmap)
    spec = torch.squeeze(spec, dim=0)
    print(f"debug_heatmap_alignment_corners {ii} {class_name=} {spec.shape=} {heatmap.shape=}")
    heatmap_up = heatmap
    heatmap_up = heatmap_up / (heatmap_up.max() + 1e-6)
    summarize_array("debug_heatmap_alignment_corners", heatmap_up)
    Hh, Wh = heatmap.shape
    Hs, Ws = spec.shape
    # Scale factors
    scale_x = Ws / Wh
    scale_y = Hs / Hh
    # Convert heatmap coords → spectrogram coords
    x1_disp = x1 * scale_x
    x2_disp = x2 * scale_x
    y1_disp = y2 * scale_y
    y2_disp = y1 * scale_y
    # Plot
    matplotlib.pyplot.figure(figsize=(20,12))
    print(f"debug_heatmap_alignment_corners {x1_disp=}, {x2_disp=} {y1_disp=}, {y2_disp=}")
    #matplotlib.pyplot.axis([x1_disp * 0.75, x2_disp * 1.25, y1_disp * 0.75, y2_disp * 1.25])
    #matplotlib.pyplot.imshow(spec, cmap='gray', alpha=0.5, origin='lower')
    matplotlib.pyplot.imshow(torch.log1p(heatmap_up), cmap='jet', vmin=0, vmax=0.1, origin='lower')
    # Draw box using corners only
    matplotlib.pyplot.plot([x1_disp, x2_disp, x2_disp, x1_disp, x1_disp],[y1_disp, y1_disp, y2_disp, y2_disp, y1_disp], 'lime' )
    matplotlib.pyplot.title(f"{ii} {class_name}")
    matplotlib.pyplot.show()"""

def draw_gaussian(heatmap, corner, sigmax, sigmay=None):
    if sigmay is None:
        sigmay = sigmax
    tmp_size = numpy.maximum(sigmax, sigmay) * 3
    mu_x = int(corner[0] + 0.5)
    mu_y = int(corner[1] + 0.5)
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

def target_heatmaps(spec_op_shape: Tuple[int, int], sampling_rate: int, ann: AnnotationGroup, class_names, spec) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, AnnotationGroup]:
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
    if len(ann_aug["individual_ids"]) == 1:
        ann_aug["individual_ids"][0] = 0
    y_2d_det = numpy.zeros((1, op_height, op_width), dtype=numpy.float32)
    y_2d_size = numpy.zeros((2, op_height, op_width), dtype=numpy.float32)
    # num classes and "background" class
    y_2d_classes: numpy.ndarray = numpy.zeros((num_classes + 1, op_height, op_width), dtype=numpy.float32)
    # create 2D ground truth heatmaps
    for ii in valid_inds:
        sigma = GAUSSIAN_SIGMA ## Try 5, 6
        draw_gaussian(y_2d_det[0, :], (x_pos_start[ii], y_pos_low[ii]), sigma)        
        y_2d_size[0, y_pos_low[ii], x_pos_start[ii]] = bb_widths[ii]
        y_2d_size[1, y_pos_low[ii], x_pos_start[ii]] = bb_heights[ii]        
        cls_id = ann["class_ids"][ii]
        if cls_id > -1:
            draw_gaussian(y_2d_classes[cls_id, :], (x_pos_start[ii], y_pos_low[ii]), sigma)
            #debug_heatmap_alignment_corners(spec, y_2d_classes[cls_id, :], x_pos_start[ii], x_pos_end[ii], y_pos_low[ii], y_pos_high[ii], ii, class_names[cls_id])
    y_2d_classes[num_classes, :] = 1.0 - y_2d_classes.sum(0)
    y_2d_classes = y_2d_classes / y_2d_classes.sum(0)[numpy.newaxis, ...]
    y_2d_classes[numpy.isnan(y_2d_classes)] = 0.0
    return y_2d_det, y_2d_size, y_2d_classes, ann_aug

def resample_audio(num_samples, sampling_rate, audio2, sampling_rate2):
    if sampling_rate != sampling_rate2:
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
    def __init__(self, data_anns_ip, class_names, is_train=False):
        self.data_anns = []
        self.audio_file = []
        self.is_train = is_train
        self.class_names = class_names
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
                if compositeClass in class_names:
                    aa["class_id"] = class_names.index(compositeClass)
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
                if compositeClass in class_names:
                    dd["class_id_file"] = class_names.index(compositeClass)
                else: 
                    print(colorama.Back.RED + f"AudioLoader __init__ class_name {compositeClass} NOT FOUND for {dd["file_path"]}" + colorama.Back.RESET)
                    dd["class_id_file"] = -1
            self.data_anns.append(dd)
            self.audio_file.append(dd["file_path"])
        ann_cnt = [len(aa["annotation"]) for aa in self.data_anns]
        self.max_num_anns = 2 * numpy.max(ann_cnt)  # x2 because we may be combining files during training
        
        self.Horseshoe_CF = {}
        for key, value in HORSESHOE_CF.items():
            id = class_names.index(key)
            self.Horseshoe_CF[id] = value
        print(f"     Num files: {len(self.data_anns)},                  Num calls: {numpy.sum(ann_cnt)}")

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
        return audio_raw, sampling_rate, duration, ann
    
    def __getitem__(self, index):
        try:
            # load audio file
            audio, sampling_rate, duration, ann = self.get_file_and_anns(index)
            if AUGMENT:
                # augment on raw audio - combine with random audio file
                if numpy.random.random() < COMBINE_PROB:
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
                if numpy.random.random() < AUG_PROB: spec = warp_spec_aug(spec, ann) 
                if numpy.random.random() < AUG_PROB: spec = mask_time_aug(spec)
                if numpy.random.random() < AUG_PROB: spec = random_bandpass_filter(spec)
                if numpy.random.random() < AUG_PROB: spec = random_time_shift(spec)
                if numpy.random.random() < AUG_PROB: 
                    if ann["class_id_file"] in self.Horseshoe_CF:
                        spec = reinforce_cf_band(spec, cf_freq=self.Horseshoe_CF[ann["class_id_file"]]) 
                    else:
                        spec = inject_vertical_noise_streak(spec)
            outputs = {}
            outputs["spec"] = spec
            # create ground truth heatmaps
            (outputs["y_2d_det"],  outputs["y_2d_size"], outputs["y_2d_classes"], ann_aug) = target_heatmaps(spec_op_shape, sampling_rate, ann, self.class_names, spec)
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

def focal_loss(pred, gt, valid_mask=None, IsClass=False):
    """ Focal loss adapted from CornerNet: Detecting Objects as Paired Keypoints
    pred  (batch x c x h x w)
    gt    (batch x c x h x w)"""
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
            per_class_loss = -per_class_neg_sum
        else:
            per_class_loss = -(per_class_pos_sum + per_class_neg_sum) / num_pos
        return per_class_loss
    return loss

def bbox_size_loss(pred_size, target_size):
    """Bounding box size loss. Only compute loss where there is a bounding box."""
    target_size_mask = (target_size > 0).float()
    return torch.nn.functional.l1_loss(pred_size * target_size_mask, target_size, reduction="sum") / (target_size_mask.sum() + 1e-5)

def loss_fun(outputs, target_det, target_size, target_class):
    detectionLoss = DET_LOSS_WEIGHT * focal_loss(outputs.pred_det, target_det)  
    boundingBoxSizeLoss = SIZE_LOSS_WEIGHT * bbox_size_loss(outputs.pred_size, target_size)
    valid_mask = (target_class[:, :-1, :, :].sum(1) > 0).float().unsqueeze(1) 
    p_class = outputs.pred_class[:, :-1, :]
    per_class_loss = focal_loss(p_class, target_class[:, :-1, :], valid_mask=valid_mask, IsClass=True)
    return detectionLoss, boundingBoxSizeLoss, per_class_loss

def build_class_weight_vector(class_names, class_weight_dict, device):
    weights = []
    for name in class_names:
        if name not in class_weight_dict:
            raise KeyError(f"Missing weight for class: {name}")
        weights.append(class_weight_dict[name])
    # Return as a torch tensor with NO grad
    return torch.tensor(weights, dtype=torch.float32).to(device)

class Trainer():
    def __init__(self, device, class_names, class_weight_vector, ip_height, len_train_loader):
        self.device = device
        self.classes = class_names
        self.num_classes =len(class_names)
        self.class_weight_vector = torch.tensor(class_weight_vector, device=self.device)
        self.min_loss = 1.79e308
        model = Net2dFast.Net2dFast(Classifier.NUM_FILTERS, num_classes=self.num_classes, ip_height=ip_height)
        self.model = model.to(Classifier.DEVICE)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, MAX_EPOCHS * len_train_loader)
        
    def train(self, epoch, data_loader):
        self.model.train()
        det_loss_sum = size_loss_sum = 0; 
        weighted_per_class_loss_sum = torch.zeros(self.num_classes).to(self.device)
        unweighted_per_class_loss_total = torch.zeros(self.num_classes).to(self.device) 
        count = 0
        epsilon = 1e-8 # prevents divide by zero problems
        for batch_idx, inputs in enumerate(data_loader):
            try:
                data = inputs["spec"].to(self.device)
                target_det = inputs["y_2d_det"].to(self.device)
                target_size = inputs["y_2d_size"].to(self.device)
                target_class = inputs["y_2d_classes"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                det_loss, size_loss, per_class_loss = loss_fun(outputs, target_det, target_size, target_class)
                weighted_per_class_loss = per_class_loss * self.class_weight_vector
                det_loss_sum += det_loss.item() * data.shape[0]; size_loss_sum += size_loss.item() * data.shape[0]
                weighted_per_class_loss_sum += weighted_per_class_loss * data.shape[0]
                unweighted_per_class_loss_total += per_class_loss * data.shape[0]
                count += data.shape[0]
                loss = det_loss + size_loss + weighted_per_class_loss.sum() 
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            except Exception as e:
                print(colorama.Back.BLUE + f"[WARNING] Skipping batch {batch_idx}: {e}" + colorama.Back.RESET)
                traceback.print_exc()
                continue
        det_loss_avg = det_loss_sum / count; size_loss_avg = size_loss_sum / count; class_loss_avg = weighted_per_class_loss_sum.sum() / count
        train_loss = det_loss_avg + size_loss_avg + class_loss_avg
        style = ""
        if train_loss < self.min_loss:
            self.best_epoch = epoch
            self.min_loss = train_loss
            style = colorama.Style.BRIGHT
            if epoch >= MIN_EPOCHS: 
                self.best_model = copy.deepcopy(self.model)
        print(style + f"{epoch=} Train loss {train_loss:.3f} = detection {det_loss_avg:.3f} + box size {size_loss_avg:.3f} + class {class_loss_avg:.3f}" + colorama.Style.RESET_ALL)
        return float(train_loss)

def main():
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    #boosted_learning_rate = False
    # setup arg parser and populate it with exiting parameters - will not work with lists
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data_dir", type=str, help="Path to root of datasets")
    parser.add_argument("model_dir",type=str,help="Directory containing trained model files.")
    args = parser.parse_args()
    if torch.cuda.is_available(): print(colorama.Fore.GREEN + "torch.cuda.is_available" + colorama.Fore.RESET)
    else: print(colorama.Fore.RED + "torch.cuda is not available" + colorama.Fore.RESET)
    model_num = next_model_number(args.model_dir)
    
    with wakepy.keep.running():
        (data_train, class_names, class_inv_freq) = load_set_of_anns(args.training_data_dir)
        model_params = dict()
        model_params["class_names"] = class_names
        loss_file = os.path.join(args.model_dir, "loss.npy")
        class_weight_vector = build_class_weight_vector(class_names, DEFAULT_CLASS_WEIGHTS, device)
    
        # train loader
        train_dataset = AudioLoader(data_train, class_names, is_train=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,)
        inputs_train = next(iter(train_loader))
        model_params["ip_height"] = ip_height = int(Classifier.SPEC_HEIGHT * Classifier.RESIZE_FACTOR)
        trainer = Trainer(device, class_names, class_weight_vector, ip_height, len(train_loader))
        # main train loop
        for epoch in range(0, MAX_EPOCHS + 1):
            train_loss = trainer.train(epoch, train_loader)
            if epoch > MIN_EPOCHS and epoch % NUM_SAVE_EPOCHS == 0:
                # save trained model
                op_state = {"epoch": trainer.best_epoch + 1, "state_dict": trainer.best_model.state_dict(), "params": model_params}
                model_file_name = f"model_{model_num}_E{trainer.best_epoch}.pth.tar"
                save_path = os.path.join(args.model_dir, model_file_name)
                torch.save(op_state, save_path)
                print(f"Saved model: {save_path}")
                if epoch - trainer.best_epoch > 50:
                    break # have plateaued
                    """if boosted_learning_rate == False:
                        boosted_learning_rate = True
                        for g in trainer.optimizer.param_groups:
                            g['lr'] *= LEARNING_RATE
                    else:
                        break # have plateaued already"""

if __name__ == "__main__":
    main()