import argparse, json, warnings, numpy, torch, datetime, os, glob, copy, torchaudio, librosa
import Net2dFast, Classifier
import ClassifierConstants as c

warnings.filterwarnings("ignore", category=UserWarning)
    
def load_set_of_anns(wav_path):
    # load the annotations
    files = glob.glob(os.path.join(wav_path, "ann", "*.json"), recursive=True)
    anns = []
    print(f"load_anns_from_path {len(files)=}")
    for ff in files:
        with open(ff) as da:
            ann = json.load(da)
        ann["file_path"] = os.path.join(wav_path, ann["id"])
        anns.append(ann)
        print(f"load_set_of_anns {ann["file_path"]=}")
    print(f"load_set_of_anns {len(anns)=}")
    # get unique class names
    class_names_all = []
    for ann in anns:
        print(f"load_set_of_anns {ann["file_path"]=}")
        for aa in ann["annotation"]:
            class_names_all.append(aa["class"] + "-" + aa["event"])
    class_names, class_cnts = numpy.unique(class_names_all, return_counts=True)
    class_inv_freq = class_cnts.sum() / (len(class_names) * class_cnts.astype(numpy.float32))
    print("Class count:")
    str_len = numpy.max([len(cc) for cc in class_names]) + 5
    for cc in range(len(class_names)):
        print(str(cc).ljust(5) + class_names[cc].ljust(str_len) + str(class_cnts[cc]))
    return anns, class_names.tolist(), class_inv_freq 

def get_params():
    params = {}
    now_str = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    model_name = now_str + ".pth.tar"
    params["model_file_name"] = model_name
    params["class_names"] = []
    return params

#batdetect2.train.audio_dataloader AudioLoader
def echo_aug(audio, sampling_rate):
    print(f"echo_aug")
    sample_offset = ( int(c.ECHO_MAX_DELAY * numpy.random.random() * sampling_rate) + 1)
    audio[:-sample_offset] += numpy.random.random() * audio[sample_offset:]
    return audio
    
def mask_time_aug(spec):
    # Mask out a random block of time - repeat up to 3 times
    print(f"mask_time_aug")
    fm = torchaudio.transforms.TimeMasking(int(spec.shape[1] * c.MASK_MAX_TIME_PERC))
    for ii in range(numpy.random.randint(1, 4)):
        spec = fm(spec)
    return spec

def mask_freq_aug(spec):
    # Mask out a random frequncy range - repeat up to 3 times
    print(f"mask_freq_aug")
    fm = torchaudio.transforms.FrequencyMasking(int(spec.shape[1] * c.MASK_MAX_FREQ_PERC))
    for ii in range(numpy.random.randint(1, 4)):
        spec = fm(spec)
    return spec

def scale_vol_aug(spec):
    print(f"scale_vol_aug")
    return spec * numpy.random.random() * c.SPEC_AMP_SCALING

def warp_spec_aug(spec, ann, return_spec_for_viz):
    # Augment spectrogram by randomly stretch and squeezing
    # NOTE this also changes the start and stop time in place
    # not taking care of spec for viz
    print(f"warp_spec_aug")
    if return_spec_for_viz:  assert False
    op_size = (spec.shape[1], spec.shape[2])
    resize_fract_r = numpy.random.rand() * c.STRETCH_SQUEEZE_DELTA * 2 - c.STRETCH_SQUEEZE_DELTA + 1.0
    resize_amt = int(spec.shape[2] * resize_fract_r)
    if resize_amt >= spec.shape[2]:
        spec_r = torch.cat((spec, torch.zeros((1, spec.shape[1], resize_amt - spec.shape[2]), dtype=spec.dtype,)), 2)
    else: spec_r = spec[:, :, :resize_amt]
    spec = torch.nn.functional.interpolate(spec_r.unsqueeze(0), size=op_size, mode="bilinear", align_corners=False).squeeze(0)
    ann["start_times"] *= 1.0 / resize_fract_r
    ann["end_times"] *= 1.0 / resize_fract_r
    return spec

def draw_gaussian(heatmap, center, sigmax, sigmay=None):
    # center is (x, y) this edits the heatmap inplace
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
    
def time_to_x_coords(time_in_file: float, samplerate: float = c.TARGET_SAMPLERATE_HZ, window_duration: float = c.FFT_WIN_LENGTH_S, window_overlap: float = c.FFT_OVERLAP) -> float:
    nfft = numpy.floor(window_duration * samplerate)  # int() uses floor
    noverlap = numpy.floor(window_overlap * nfft)
    return (time_in_file * samplerate - noverlap) / (nfft - noverlap)

def ground_truth_heatmaps(spec_op_shape: Tuple[int, int], sampling_rate: int, ann: AnnotationGroup, class_names) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, AnnotationGroup]:
    # spec may be resized on input into the network
    num_classes = len(class_names)
    op_height = spec_op_shape[0]
    op_width = spec_op_shape[1]
    freq_per_bin = (c.MAX_FREQ_HZ - c.MIN_FREQ_HZ) / op_height

    # start and end times
    x_pos_start = time_to_x_coords(ann["start_times"], sampling_rate, c.FFT_WIN_LENGTH_S, c.FFT_OVERLAP)
    x_pos_start = (c.RESIZE_FACTOR * x_pos_start).astype(numpy.int32)
    x_pos_end = time_to_x_coords(ann["end_times"], sampling_rate, c.FFT_WIN_LENGTH_S, c.FFT_OVERLAP)
    x_pos_end = (c.RESIZE_FACTOR * x_pos_end).astype(numpy.int32)

    # location on y axis i.e. frequency
    y_pos_low = (ann["low_freqs"] - c.MIN_FREQ_HZ) / freq_per_bin
    y_pos_low = (op_height - y_pos_low).astype(numpy.int32)
    y_pos_high = (ann["high_freqs"] - c.MIN_FREQ_HZ) / freq_per_bin
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
        draw_gaussian(y_2d_det[0, :], (x_pos_start[ii], y_pos_low[ii]), c.TARGET_SIGMA)
        y_2d_size[0, y_pos_low[ii], x_pos_start[ii]] = bb_widths[ii]
        y_2d_size[1, y_pos_low[ii], x_pos_start[ii]] = bb_heights[ii]

        cls_id = ann["class_ids"][ii]
        if cls_id > -1:
            draw_gaussian(y_2d_classes[cls_id, :], (x_pos_start[ii], y_pos_low[ii]), c.TARGET_SIGMA)

    # be careful as this will have a 1.0 places where we have event but dont know gt class this will be masked in training anyway
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

def pad_array(ip_array, pad_size):
    return numpy.hstack((ip_array, numpy.ones(pad_size, dtype=numpy.int32) * -1))

class AudioLoader(torch.utils.data.Dataset):
    def __init__(self, data_anns_ip, params, dataset_name=None, is_train=False):
        self.data_anns = []
        self.is_train = is_train
        self.params = params
        self.return_spec_for_viz = False
        print(f"AudioLoader __init__ {len(data_anns_ip)=}")
        for ii in range(len(data_anns_ip)):
            dd = copy.deepcopy(data_anns_ip[ii])
            # filter out unused annotation here
            filtered_annotations = []
            for ii, aa in enumerate(dd["annotation"]):  #### list indices must be integers or slices, not str ####
                if "individual" in aa.keys():
                    aa["individual"] = int(aa["individual"])
                    # if only one call labeled it has to be from the same individual
                    if len(dd["annotation"]) == 1:
                        aa["individual"] = 0
                # convert class name into class label
                if aa["class"] in self.params["class_names"]:
                    print(f"AudioLoader __init__ {params["class_names"]=} {aa["class"]=}")
                    aa["class_id"] = self.params["class_names"].index(aa["class"])
                else:
                    aa["class_id"] = -1
                filtered_annotations.append(aa)

            dd["annotation"] = filtered_annotations
            dd["start_times"] = numpy.array([aa["start_time"] for aa in dd["annotation"]])
            dd["end_times"] = numpy.array([aa["end_time"] for aa in dd["annotation"]])
            dd["high_freqs"] = numpy.array([float(aa["high_freq"]) for aa in dd["annotation"]])
            dd["low_freqs"] = numpy.array([float(aa["low_freq"]) for aa in dd["annotation"]])
            dd["class_ids"] = numpy.array([aa["class_id"] for aa in dd["annotation"]]).astype(numpy.int32)
            dd["individual_ids"] = numpy.array([aa["individual"] for aa in dd["annotation"]]).astype(numpy.int32)

            # file level class name
            dd["class_id_file"] = -1
            print(f" {dd.keys()=} {self.params["class_names"]=}")
            if "class_name" in dd.keys():
                if dd["class_name"] in self.params["class_names"]:
                    dd["class_id_file"] = self.params["class_names"].index(dd["class_name"])

            self.data_anns.append(dd)

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

        audio_file = self.data_anns[index]["file_path"]
        sampling_rate, audio_raw = Classifier.load_audio(audio_file, self.data_anns[index]["time_exp"], c.TARGET_SAMPLERATE_HZ)

        # copy annotation
        ann = {}
        ann["annotated"] = self.data_anns[index]["annotated"]
        ann["class_id_file"] = self.data_anns[index]["class_id_file"]
        keys = ["start_times", "end_times", "high_freqs", "low_freqs", "class_ids", "individual_ids"]
        for kk in keys:
            ann[kk] = self.data_anns[index][kk].copy()

        # if train then grab a random crop
        nfft = c.FFT_WIN_LENGTH_S * sampling_rate
        noverlap = c.FFT_OVERLAP * nfft
        length_samples = int(c.SPEC_TRAIN_WIDTH * (nfft - noverlap) + noverlap)
        if audio_raw.shape[0] - length_samples > 0:
            sample_crop = numpy.random.randint(audio_raw.shape[0] - length_samples)
        else:
            sample_crop = 0
        print(f"get_file_and_anns {sample_crop=} {length_samples=} {len(audio_raw)=}")
        audio_raw = audio_raw[sample_crop : sample_crop + length_samples]
        ann["start_times"] = ann["start_times"] - sample_crop / float(sampling_rate)
        ann["end_times"] = ann["end_times"] - sample_crop / float(sampling_rate)

        # pad audio
        op_spec_target_size = c.SPEC_TRAIN_WIDTH
        audio_raw = Classifier.pad_audio(audio_raw, sampling_rate, c. FFT_WIN_LENGTH_S, c.FFT_OVERLAP, c.RESIZE_FACTOR, c.SPEC_DIVIDE_FACTOR, op_spec_target_size,)
        duration = audio_raw.shape[0] / float(sampling_rate)

        # sort based on time
        inds = numpy.argsort(ann["start_times"])
        for kk in ann.keys():
            if (kk != "class_id_file") and (kk != "annotated"):
                ann[kk] = ann[kk][inds]

        return audio_raw, sampling_rate, duration, ann
    
    def __getitem__(self, index):

        # load audio file
        audio, sampling_rate, duration, ann = self.get_file_and_anns(index)

        # augment on raw audio
        # augment - combine with random audio file
        if numpy.random.random() < c.AUG_PROB:
            (audio2, sampling_rate2, duration2, ann2) = self.get_file_and_anns()
            audio, ann = combine_audio_aug(audio, sampling_rate, ann, audio2, sampling_rate2, ann2)

        # simulate echo by adding delayed copy of the file
        if numpy.random.random() < c.AUG_PROB:
            audio = echo_aug(audio, sampling_rate)

        # create spectrogram
        spec, spec_for_viz = Classifier.generate_spectrogram(audio, sampling_rate, self.return_spec_for_viz)
        spec_op_shape = (int(c.SPEC_HEIGHT * c.RESIZE_FACTOR), int(spec.shape[1] * c.RESIZE_FACTOR))

        # resize the spec
        spec = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
        spec = torch.nn.functional.interpolate(spec, size=spec_op_shape, mode="bilinear", align_corners=False).squeeze(0)

        # augment spectrogram
        if numpy.random.random() < c.AUG_PROB: spec = scale_vol_aug(spec)
        if numpy.random.random() < c.AUG_PROB: spec = warp_spec_aug(spec, ann, self.return_spec_for_viz)
        if numpy.random.random() < c.AUG_PROB: spec = mask_time_aug(spec)
        if numpy.random.random() < c.AUG_PROB: spec = mask_freq_aug(spec)
        outputs = {}
        outputs["spec"] = spec
        if self.return_spec_for_viz:
            outputs["spec_for_viz"] = torch.from_numpy(spec_for_viz).unsqueeze(0)

        # create ground truth heatmaps
        (outputs["y_2d_det"],  outputs["y_2d_size"], outputs["y_2d_classes"], ann_aug) = ground_truth_heatmaps(spec_op_shape, sampling_rate, ann, self.params["class_names"])

        # hack to get around requirement that all vectors are the same length in the output batch
        pad_size = self.max_num_anns - len(ann_aug["individual_ids"])
        outputs["is_valid"] = pad_array(numpy.ones(len(ann_aug["individual_ids"])), pad_size)
        keys = ["class_ids", "individual_ids", "x_inds", "y_inds", "start_times",  "end_times", "low_freqs", "high_freqs"]
        for kk in keys:
            outputs[kk] = pad_array(ann_aug[kk], pad_size)

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

    def __len__(self):
        return len(self.data_anns)

def compute_pre_rec(gts, preds, eval_mode, class_of_interest, num_classes, threshold,  ignore_start_end):
    """Computes precision and recall. Assumes that each file has been exhaustively
    annotated. Will not count predicted detection with a start time that is within
    ignore_start_end miliseconds of the start or end of the file.
    eval_mode == 'detection'
      Returns overall detection results (not per class)
    eval_mode == 'per_class'
      Filters ground truth based on class of interest. This will ignore predictions
      assigned to gt with unknown class.
    eval_mode = 'top_class'
       Turns the problem into a binary one and selects the top predicted class
       for each predicted detection"""

    # get predictions and put in array
    pred_boxes = []
    confidence = []
    pred_class = []
    file_ids = []
    for pid, pp in enumerate(preds):

        # filter predicted calls that are too near the start or end of the file
        file_dur = gts[pid]["duration"]
        valid_inds = (pp["start_times"] >= ignore_start_end) & (pp["start_times"] <= (file_dur - ignore_start_end))

        pred_boxes.append(numpy.vstack((pp["start_times"][valid_inds], pp["end_times"][valid_inds], pp["low_freqs"][valid_inds],  pp["high_freqs"][valid_inds])).T)

        if eval_mode == "detection":
            # overall detection
            confidence.append(pp["det_probs"][valid_inds])
        elif eval_mode == "per_class":
            # per class
            confidence.append(pp["class_probs"].T[valid_inds, class_of_interest])
        elif eval_mode == "top_class":
            # per class - note that sometimes 'class_probs' can be num_classes+1 in size
            top_class = numpy.argmax(pp["class_probs"].T[valid_inds, :num_classes], 1)
            confidence.append(pp["class_probs"].T[valid_inds, top_class])
            pred_class.append(top_class)

        # be careful, assuming the order in the list is same as GT
        file_ids.append([pid] * valid_inds.sum())

    confidence = numpy.hstack(confidence)
    file_ids = numpy.hstack(file_ids).astype(numpy.int32)
    pred_boxes = numpy.vstack(pred_boxes)
    if len(pred_class) > 0:
        pred_class = numpy.hstack(pred_class)

    # extract relevant ground truth boxes
    gt_boxes = []
    gt_assigned = []
    gt_class = []
    gt_generic_class = []
    num_positives = 0
    for gg in gts:
        # filter ground truth calls that are too near the start or end of the file
        file_dur = gg["duration"]
        valid_inds = (gg["start_times"] >= ignore_start_end) & (gg["start_times"] <= (file_dur - ignore_start_end))

        # note, files with the incorrect duration will cause a problem
        if (gg["start_times"] > file_dur).sum() > 0:
            print("Error: file duration incorrect for", gg["id"])
            assert False

        boxes = numpy.vstack((gg["start_times"][valid_inds], gg["end_times"][valid_inds],  gg["low_freqs"][valid_inds],  gg["high_freqs"][valid_inds])).T
        gen_class = gg["class_ids"][valid_inds] == -1
        class_ids = gg["class_ids"][valid_inds]

        # keep track of the number of relevant ground truth calls
        if eval_mode == "detection":
            # all valid ones
            num_positives += len(gg["start_times"][valid_inds])
        elif eval_mode == "per_class":
            # all valid ones with class of interest
            num_positives += (gg["class_ids"][valid_inds] == class_of_interest).sum()
        elif eval_mode == "top_class":
            # all valid ones with non generic class
            num_positives += (gg["class_ids"][valid_inds] > -1).sum()

        # find relevant classes (i.e. class_of_interest) and events without known class (i.e. generic class, -1)
        if eval_mode == "per_class":
            class_inds = (class_ids == class_of_interest) | (class_ids == -1)
            boxes = boxes[class_inds, :]
            gen_class = gen_class[class_inds]
            class_ids = class_ids[class_inds]

        gt_assigned.append(numpy.zeros(boxes.shape[0]))
        gt_boxes.append(boxes)
        gt_generic_class.append(gen_class)
        gt_class.append(class_ids)

    # loop through detections and keep track of those that have been assigned
    true_pos = numpy.zeros(confidence.shape[0])
    valid_inds = numpy.ones(confidence.shape[0]) == 1  # intialize to True
    sorted_inds = numpy.argsort(confidence)[::-1]  # sort high to low
    for ii, ind in enumerate(sorted_inds):
        gt_id = file_ids[ind]
        valid_det = False
        if gt_boxes[gt_id].shape[0] > 0:
            # compute overlap
            valid_det, det_ind = compute_affinity_1d(pred_boxes[ind], gt_boxes[gt_id], threshold)

        # valid detection that has not already been assigned
        if valid_det and (gt_assigned[gt_id][det_ind] == 0):

            count_as_true_pos = True
            if eval_mode == "top_class" and (gt_class[gt_id][det_ind] != pred_class[ind]):
                # needs to be the same class
                count_as_true_pos = False

            if count_as_true_pos:
                true_pos[ii] = 1

            gt_assigned[gt_id][det_ind] = 1

            # if event is generic class (i.e. gt_generic_class[gt_id][det_ind] is True)
            # and eval_mode != 'detection', then ignore it
            if gt_generic_class[gt_id][det_ind]:
                if eval_mode == "per_class" or eval_mode == "top_class":
                    valid_inds[ii] = False

    # store threshold values - used for plotting
    conf_sorted = numpy.sort(confidence)[::-1][valid_inds]
    thresholds = numpy.linspace(0.1, 0.9, 9)
    thresholds_inds = numpy.zeros(len(thresholds), dtype=numpy.int32)
    for ii, tt in enumerate(thresholds):
        thresholds_inds[ii] = numpy.argmin(conf_sorted > tt)
    thresholds_inds[thresholds_inds == 0] = -1

    # compute precision and recall
    true_pos = true_pos[valid_inds]
    false_pos_c = numpy.cumsum(1 - true_pos)
    true_pos_c = numpy.cumsum(true_pos)

    recall = true_pos_c / num_positives
    precision = true_pos_c / numpy.maximum(true_pos_c + false_pos_c, numpy.finfo(numpy.float64).eps)

    results = {}
    results["recall"] = recall
    results["precision"] = precision
    results["num_gt"] = num_positives

    results["thresholds"] = thresholds
    results["thresholds_inds"] = thresholds_inds

    if num_positives == 0:
        results["avg_prec"] = numpy.nan
        results["rec_at_x"] = numpy.nan
    else:
        results["avg_prec"] = numpy.round(calc_average_precision(recall, precision), 5)
        results["rec_at_x"] = numpy.round(calc_recall_at_x(recall, precision), 5)
    return results

def evaluate_predictions(gts, preds, class_names, detection_overlap, ignore_start_end=0.0):
    """ Computes metrics derived from the precision and recall.
    Assumes that gts and preds are both lists of the same lengths, with ground
    truth and predictions contained within.
    Returns the overall detection results, and per class results"""
    assert len(gts) == len(preds)
    num_classes = len(class_names)

    # evaluate detection on its own i.e. ignoring class
    det_results = compute_pre_rec(gts, preds, "detection", None, num_classes, detection_overlap, ignore_start_end,)
    top_class = compute_pre_rec(gts, preds, "top_class", None, num_classes, detection_overlap, ignore_start_end)
    det_results["top_class"] = top_class

    # per class evaluation
    det_results["class_pr"] = []
    for cc in range(num_classes):
        res = compute_pre_rec(gts, preds, "per_class", cc,  num_classes, detection_overlap, ignore_start_end)
        res["name"] = class_names[cc]
        det_results["class_pr"].append(res)

    # ignores classes that are not present in the test set
    det_results["avg_prec_class"] = numpy.mean([rs["avg_prec"] for rs in det_results["class_pr"] if rs["num_gt"] > 0])
    det_results["avg_prec_class"] = numpy.round(det_results["avg_prec_class"], 5)

    # file level evaluation
    res_file = compute_file_accuracy(gts, preds, num_classes)
    det_results.update(res_file)

    return det_results

def focal_loss(pred, gt, weights=None, valid_mask=None):
    """ Focal loss adapted from CornerNet: Detecting Objects as Paired Keypoints
    pred  (batch x c x h x w)
    gt    (batch x c x h x w)"""
    eps = 1e-5
    beta = 4
    alpha = 2
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = (torch.log(1 - pred + eps) * torch.pow(pred, alpha) * torch.pow(1 - gt, beta) * neg_inds)
    if weights is not None:
        pos_loss = pos_loss * weights
    if valid_mask is not None:
        pos_loss = pos_loss * valid_mask
        neg_loss = neg_loss * valid_mask
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def bbox_size_loss(pred_size, gt_size):
    """Bounding box size loss. Only compute loss where there is a bounding box."""
    gt_size_mask = (gt_size > 0).float()
    return torch.nn.functional.l1_loss(pred_size * gt_size_mask, gt_size, reduction="sum") / (gt_size_mask.sum() + 1e-5)

def loss_fun(outputs, gt_det, gt_size, gt_class, class_inv_freq):
    detectionLoss = c.DET_LOSS_WEIGHT * focal_loss(outputs.pred_det, gt_det)  
    boundingBoxSizeLoss = c.SIZE_LOSS_WEIGHT * bbox_size_loss(outputs.pred_size, gt_size)
    valid_mask = (gt_class[:, :-1, :, :].sum(1) > 0).float().unsqueeze(1)
    p_class = outputs.pred_class[:, :-1, :]
    classLoss = c.CLASS_LOSS_WEIGHT * focal_loss(p_class, gt_class[:, :-1, :], valid_mask=valid_mask)
    print(f"loss_fun {gt_class=} {class_inv_freq=} {gt_size=} {gt_det=} {detectionLoss=} {boundingBoxSizeLoss=} {classLoss=}")
    return detectionLoss + boundingBoxSizeLoss + classLoss

def train(model, epoch, data_loader, optimizer, scheduler, params):
    model.train()
    class_inv_freq = torch.from_numpy(numpy.array(params["class_inv_freq"], dtype=numpy.float32)).to(params["device"])
    class_inv_freq = class_inv_freq.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    print("\nEpoch", epoch)
    sum = 0; 
    count = 0
    for batch_idx, inputs in enumerate(data_loader):
        data = inputs["spec"].to(params["device"])
        gt_det = inputs["y_2d_det"].to(params["device"])
        gt_size = inputs["y_2d_size"].to(params["device"])
        gt_class = inputs["y_2d_classes"].to(params["device"])
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fun(outputs, gt_det, gt_size, gt_class, class_inv_freq)
        sum += loss.item() * data.shape[0]
        count += data.shape[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % 50 == 0 and batch_idx != 0:
            print("[{}/{}]\tLoss: {:.4f}".format(batch_idx * len(data), len(data_loader.dataset), sum / count))
    print("Train loss          : {:.4f}".format(sum / count))
    res = {}
    res["train_loss"] = float(sum / count)
    return res

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
    (data_train, params["class_names"], class_inv_freq) = load_set_of_anns(params["data_dir"])
    params["class_inv_freq"] = class_inv_freq.tolist()
    
    # train loader
    train_dataset = AudioLoader(data_train, params, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True, num_workers=c.NUM_WORKERS, pin_memory=True,)
    inputs_train = next(iter(train_loader))
    params["ip_height"] = int(c.SPEC_HEIGHT * c.RESIZE_FACTOR)
    print("\ntrain batch spec size :", inputs_train["spec"].shape)
    print("class target size     :", inputs_train["y_2d_classes"].shape)

    # select network
    num_classes = len(params["class_names"])
    model = Net2dFast.Net2dFast(c.NUM_FILTERS,num_classes=num_classes, ip_height=params["ip_height"])
    model = model.to(c.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=c.LR) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c.NUM_EPOCHS * len(train_loader))

    # save parameters to file 
    with open(os.path.join(params["data_dir"], "params.json"), "w") as da:
        json.dump(params, da, indent=2, sort_keys=True)
        
    # main train loop
    for epoch in range(0, c.NUM_EPOCHS + 1):
        train_loss = train(model, epoch, train_loader, optimizer,  scheduler, params)

    # save trained model
    print("saving model to: " + params["model_file_name"])
    op_state = {"epoch": epoch + 1, "state_dict": model.state_dict(), "params": params}
    torch.save(op_state, os.path.join(params["data_dir"], params["model_file_name"]))

if __name__ == "__main__":
    main()
