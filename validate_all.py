import argparse, warnings, numpy, torch, datetime, os, glob, copy, polars, collections
import torchaudio, librosa, traceback, colorama, inspect, wakepy, random, math, scipy
import Net2dFast, Classifier, validate_model, training_debug_logger

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(threshold=torch.inf, linewidth=200, precision=3)
numpy.set_printoptions(threshold=numpy.inf)
numpy.set_printoptions(precision=4, suppress=True)

def main():
    global CONSISTENCY_LOSS_WEIGHT, MAX_EPOCHS
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    #boosted_learning_rate = False
    # setup arg parser and populate it with exiting parameters - will not work with lists
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_data_dir", type=str, help="Path to the root directory of the validation dataset.")
    parser.add_argument("model",type=str,help="Directory for trained model files, or model to be refined")
    args = parser.parse_args()
    if torch.cuda.is_available(): print(colorama.Fore.GREEN + "torch.cuda.is_available" + colorama.Fore.RESET)
    else: print(colorama.Fore.RED + "torch.cuda is not available" + colorama.Fore.RESET)
    
    model_dir = args.model
    with wakepy.keep.running():

        models = glob.glob(os.path.join(model_dir, "**", "*.pth.tar"), recursive=False)
        for model_path in models:
            f1_score = validate_model.validate_model(model_path, args.validation_data_dir , writeFile=False)
if __name__ == "__main__":
    main()