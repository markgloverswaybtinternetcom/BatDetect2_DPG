import os, pandas, wakepy, colorama
from Classifier import Classify
import argparse, utils

def FileDrop(f):
	if os.path.isdir(f):
		dirResults_file = os.path.join(f, "BatDetect2 Results.csv")
		if os.path.isfile(dirResults_file):
			print(f"FileDrop {dirResults_file=} found")
			print(f"{f} files already Classified")
		else:
			ClassifyDir(f)
	elif os.path.isfile(f):
		dir = os.path.dirname(f); file = os.path.basename(f); callsCsvPath = os.path.join(dir, "ann", file + ".csv")
		if not os.path.isfile(callsCsvPath):
			print(f"LoadClassifiedFile {callsCsvPath=} not found")
			self.classify.File(f, debug=True)
			print(f"Classified file {f}")
		else: print("NO FILE OR DIRECTOY")

def ClassifyDir(dir_path):
    with wakepy.keep.running():
        classify = Classify()
        files = utils.ListAudioFiles(dir_path)
        FilesDF = pandas.DataFrame(columns =["Filename", "Bat Calls"])
        length = len(files)
        for index, audio_file in enumerate(files): 
            result = classify.File(audio_file)
            file = os.path.basename(audio_file)
            if len(result) > 0: 
                r = len(FilesDF)
                FilesDF.loc[r] = [file, result]			
            print(f"file {index +1} of {length} Classified", end='\r') 
        dirResults_file = os.path.join(dir_path, "BatDetect2 Results.csv")
        FilesDF.to_csv(dirResults_file, index=False)
        print(colorama.Back.GREEN + f"'{dir_path}' files are all Classified" + colorama.Back.RESET) 

parser = argparse.ArgumentParser(description="Bat call classifier")
parser.add_argument("Pathname", help="The directory or filename that needs clasifying.")
args = parser.parse_args()
FileDrop(args.Pathname)


