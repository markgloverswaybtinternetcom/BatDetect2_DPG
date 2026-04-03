import pandas, os, sys, torch, colorama, json, utils, time, soundfile, batdetect2
from typing import Any, Union
MIN_PROB = 0.2

class Classifier():
    """Uses BatDetect2 lower level code without modification any modifications are in this class"""
    def __init__(self): 
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

    def GetDfSummary(self, calls):
        """Creates file species summary info"""
        summaryDict = {}
        for call in jsonData['annotation']:
            if call["event"] == 'Echolocation': id = call["class"]
            else: id = call["class"] + "," +  call["event"]
            prob = float(call["det_prob"]) * float(call["class_prob"])
            if id in summaryDict:
                min = summaryDict[id][1]; max = summaryDict[id][2];
                if prob < min: min = prob
                if prob > max: max = prob
                summaryDict[id] = [summaryDict[id][0]+1, min, max]
            else:
                summaryDict[id] = [1, prob, prob]
        summary = ""
        for id, val in summaryDict.items():
            call_type = ""
            if id.contains(','): 
                words = id.split(',')
                latin = word[0];
                call_type = " " + word[1]
            else: latin = id
            if latin == "Barbastellus barbastellus": latin = "Barbastella barbastellus" #batdetect2 latine error
            if self.latinToLangDict is None: species = latin
            else: species = self.latinToLangDict[latin]
            if val[0] == 1: summary += f"{species}{call_type} 1 call {val[1]:.0%}, "
            else: summary += f"{species}{call_type} {val[0]} calls {val[2]:.0%}-{val[1]:.0%}, "
        return summary

    def File(self, filepath, debug=False):
        """Classifies one file using BatDetect2"""
        dir = os.path.dirname(filepath)
        file = os.path.basename(filepath)
        op_dir = os.path.join(dir,"ann")
        if not os.path.isdir(op_dir): # make directory if it does not exist
            print("Creating directory for annotation files", op_dir)
            os.makedirs(op_dir)
        op_path = os.path.join(op_dir, file + ".csv")
        try:
            calls = batdetect2.api.process_file(filepath)
            with open(callsJsonPath, "w", encoding="utf-8") as jsonfile:
                json.dump(results, jsonfile, indent=2, sort_keys=True)
        except  Exception as error:
            print(colorama.Fore.RED + f"Classifier process_file {error}" + colorama.Fore.RESET)
            summary = ""; calls = ""
        summary = self.GetDfSummary(calls) # empty file saves trying to classify again
        if len(summary)> 0: print(colorama.Fore.GREEN + f"{file}, {summary}  " + colorama.Fore.RESET, flush=True)
        return summary