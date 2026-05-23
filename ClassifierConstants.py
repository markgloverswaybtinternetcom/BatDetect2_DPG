########## constants ############
import torch

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

AUG_PROB = 0.20
DETECTION_OVERLAP = 0.01  # has to be within this number of ms to count as detection
DET_LOSS_WEIGHT = 1.0  # weight for the detection part of the loss
SIZE_LOSS_WEIGHT = 0.1  # weight for the bbox size loss
CLASS_LOSS_WEIGHT  = 2.0  # weight for the classification loss	
LR = 0.001
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 200
NUM_FILTERS = 128
SPEC_TRAIN_WIDTH = 512
TARGET_SIGMA = 2.0
ECHO_MAX_DELAY = 0.005  # simulate echo by adding copy of raw audio
STRETCH_SQUEEZE_DELTA = 0.04  # stretch or squeeze spec
MASK_MAX_TIME_PERC = 0.05  # max mask size - here percentage, not ideal
MASK_MAX_FREQ_PERC = 0.10  # max mask size - here percentage, not ideal
SPEC_AMP_SCALING = 2.0 