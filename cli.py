import os, polars, wakepy, colorama
from Classifier import Classifier
import argparse, utils
"""Colour coded Command line run classifier using BatDetect2"""

def FileDrop(f):
    """Classify a directory or file"""
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
    """Classify a directory, this may take some time so stop sleeping"""
    with wakepy.keep.running():
        classify = Classifier()
        files = utils.ListAudioFiles(dir_path, TimeExpanded=False)
        FilesDF = polars.DataFrame(schema=[("Filename", polars.Utf8), ("Bat Call", polars.Utf8)]) # Utf8 = string
        length = len(files)
        for index, audio_file in enumerate(files): 
            result = classify.File(audio_file)
            file = os.path.basename(audio_file)
            if len(result) > 0: 
                r = len(FilesDF)
                new_row = polars.DataFrame({"Filename": [file], "Bat Call": [result]})
                FilesDF = FilesDF.extend(new_row)                
            print(f"file {index +1} of {length} Classified", end='\r')
        # create a summary file
        dirResults_file = os.path.join(dir_path, "BatDetect2 Results.csv")
        FilesDF.write_csv(dirResults_file)
        print(colorama.Back.GREEN + colorama.Fore.BLACK + f"'{dir_path}' files are all Classified" + colorama.Style.RESET_ALL) 

parser = argparse.ArgumentParser(description="Bat call classifier")
parser.add_argument("Pathname", help="The directory or filename that needs clasifying.")
args = parser.parse_args()
FileDrop(args.Pathname)


