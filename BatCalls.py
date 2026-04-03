import dearpygui.dearpygui as dpg
import numpy, json, csv

class BatCalls(): 
    """Annotation information on bat call, uses numpy rather than pandas for speed"""
    
    def __init__(self, parentSelf):
        self.SpeciesNames = parentSelf.SpeciesNames
        self.SpeciesLanguage = parentSelf.SpeciesLanguage
        self.CallTypes = parentSelf.CallTypes
        self.LatinIdx = dict(zip(self.SpeciesNames["Latin"], self.SpeciesNames.index)) 
        self.CallsNP = None
        
    def fromCSV(self, callsCsvPath):
        """"Load annotation information from old BatDetect2 classifier CSV file and generate summay of contents"""
        with open(callsCsvPath, mode ='r')as file:
            csvLines = csv.reader(file)
            summaryDict = {}
            arr = []
            i = 0;
            for row in csvLines: 
                if i>0: # first line column titles
                    species = row[6]
                    if species == "Barbastellus barbastellus": species = "Barbastella barbastellus" #batdetect2 latine error
                    id = self.LatinIdx[species]; p1 = float(row[1]); p2 = float(row[7])
                    if len(row) > 8: ct = self.CallTypes.index(row[8])
                    else: ct = 0
                    arr.append([id, float(row[2]), float(row[3]), float(row[5])/1000, float(row[4])/1000, p1, p2, ct])
                    # no call type / event in standard CSV
                    prob = p1 * p2
                    if id in summaryDict:
                        min = summaryDict[id][1]; max = summaryDict[id][2];
                        if prob < min: min = prob
                        if prob > max: max = prob
                        summaryDict[id] = [summaryDict[id][0]+1, min, max]
                    else:
                        summaryDict[id] = [1, prob, prob] 
                i += 1
        summary = ""
        for id, val in summaryDict.items():
            species = self.SpeciesNames.loc[id][self.SpeciesLanguage]
            if val[0] == 1: summary += f"{species} 1 call {val[1]:.0%}, "
            else: summary += f"{species} {val[0]} calls {val[2]:.0%}-{val[1]:.0%}, "
        self.CallsNP = numpy.array(arr, dtype='f')
        return summary

    def DisplayAnnotations(self, plot, minT, maxT):
        """"Displays annotations on a spectrogram as a label"""
        dpg.delete_item(plot, children_only=True, slot=0) # remove annotations
        species = ""
        exception = ""
        if self.CallsNP is not None and self.CallsNP.shape[0] > 0:
            i = 0; abbrev= ""
            calls = self.CallsNP[numpy.where((self.CallsNP[:,1] > minT) & (self.CallsNP[:,2] < maxT) )]
            if len(calls) < 35:
                #print(f"DisplayAnnotations {minT=} {maxT=} {calls=} {self.CallsNP[:,1]=}")
                for call in calls:
                    id=int(call[0]); t1=call[1]; t2=call[2]; f1=call[3]; f2=call[4]; p1 = call[5]; p2 = call[6]; ct = int(call[7])
                    #print(f"DisplayAnnotations {id=} {t1=} {t2=} {f1=} {f2=} {p1=} {p2=} {ct=}")
                    if t2 - t1 < 0.01: t = t2 # end of FM calls
                    else: t = (t1 + t2) / 2 # mid way long or constant frequency calls 
                        
                    if self.SpeciesLanguage != "None":
                        species =  self.SpeciesNames.loc[id][self.SpeciesLanguage]
                        if ct > 0: l = species.replace(" ","\n") + '\n' + self.CallTypes[ct]
                        else: l = species.replace(" ","\n")

                        if t1 > minT:
                            if t2 > maxT: break
                            if i % 2 == 0: #same frequency = alternate rows of labels
                                ann = dpg.add_plot_annotation(parent=plot, label=l, default_value=(t, f1), offset=(0, 30), color=[150, 150, 150, 128])
                            else:
                                ann = dpg.add_plot_annotation(parent=plot, label=l, default_value=(t, f1), offset=(0, 45), color=[150, 150, 150, 128])
                            i = i +1
                        #print(f"DisplayAnnotations added {i} plot_annotations")
            else:
                exception = f"{len(calls)} calls - Too many to label"
        return species, exception

    def fromJSON(self, filepath):
        """"Load annotation information from a JSON file in format that the BatDetect2 uses for traiing models and generate summay of contents"""
        print(f"fromJSON {filepath=}")
        with open(filepath, 'r') as file:
            jsonData = json.load(file)
            summaryDict = {}
            self.CallsNP = None
            for call in jsonData['annotation']:
                id = self.LatinIdx[call["class"]]; ct = self.CallTypes.index(call["event"]); t1 = float(call["start_time"]); t2 = float(call["end_time"]); 
                f1 = float(call["low_freq"])/1000; f2 = float(call["high_freq"])/1000; p1 = float(call["det_prob"]); p2 = float(call["class_prob"])
                if self.CallsNP is None: 
                    self.CallsNP = numpy.array([[ id, t1, t2, f1, f2, p1, p2, ct]], dtype=numpy.float32) 
                else:
                    self.CallsNP = numpy.append(self.CallsNP, numpy.array([[ id, t1, t2, f1, f2, p1, p2, ct]], dtype=numpy.float32), axis=0)
                prob = p1 * p2
                if id in summaryDict:
                    min = summaryDict[id][1]; max = summaryDict[id][2];
                    if prob < min: min = prob
                    if prob > max: max = prob
                    summaryDict[id] = [summaryDict[id][0]+1, min, max]
                else:
                    summaryDict[id] = [1, prob, prob] 
                        
        summary = ""
        for id, val in summaryDict.items():
            species = self.SpeciesNames.loc[id][self.SpeciesLanguage]
            if val[0] == 1: summary += f"{species} 1 call {val[1]:.0%}, "
            else: summary += f"{species} {val[0]} calls {val[2]:.0%}-{val[1]:.0%}, "
        self.CallsNP = numpy.array(arr, dtype='f')
        return summary
                
    def toJSON(self, callsJsonPath):
        """"Save annotation information to a JSON file in format that the BatDetect2 uses for traiing models"""
        print(f"CallNPtoJSON {callsJsonPath=}")
        annotationValues = []
        for call in self.CallsNP:
            id = self.SpeciesNames["Latin"][int(call[0])]; t1=f"{call[1]:.4f}"; t2=f"{call[2]:.4f}"; f"{call[3]*1000:.0f}"
            f2=f"{call[4]*1000:.0f}"; p1 = f"{call[5]:.3f}"; p2 = f"{call[6]:.3f}"; ct = self.CallTypes.index(int(call[7]))
            c = self.SpeciesNames["Latin"][id]; callType = self.CallTypes[ct]
            annotationValues.append({'class': c, 'class_prob': prob, 'det_prob': prob, 'end_time': t2, 'event': callType, 'high_freq': f2, 'individual': '-1','low_freq': f1, 'start_time': t1})
        thisdict = {"annotated": False, "annotation": annotationValues, "class_name": c, "duration": self.duration, "id": self.file, "issued": False, "notes": "Automatically generated.", "time_exp": 1}
        with open(callsJsonPath, "w", encoding="utf-8") as jsonfile:
            json.dump(thisdict, jsonfile, indent=2, sort_keys=True)

    def toCSV(self, csvFilePath):
        """"Save annotation information to BatDetect2 classifier CSV file, after manual editing"""
        print(f"toCSV {csvFilePath=}")
        data = [['id','det_prob','start_time','end_time','high_freq','low_freq','class','class_prob','event']]
        n = 0
        for call in self.CallsNP:
            s = self.SpeciesNames["Latin"][int(call[0])]; t1=f"{call[1]:.4f}"; t2=f"{call[2]:.4f}"; f1=f"{call[3]*1000:.0f}"
            f2=f"{call[4]*1000:.0f}"; p1 = f"{call[5]:.3f}"; p2 = f"{call[6]:.3f}"; ct = self.CallTypes[int(call[7])]
            data.append([n, p1, t1, t2, f2, f1, s, p2, ct])
            n += 1
        with open(csvFilePath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
    
    def Insert(self, speciesId, callTypeId, callTmin, callTmax, callFmin, callFmax):
        print(f"Insert {speciesId=} {callTypeId=} {callTmin=} {callTmax=} {callFmin=} {callFmax=}")
        callInserted = False
        arr = []
        if self.CallsNP is not None:
            for call in self.CallsNP:
                id=int(call[0]); t1=call[1]; t2=call[2]; f1=call[3]; f2=call[4]; p1=call[5]; p2=call[6]; ct=int(call[7])
                if not callInserted and callTmax < t1: 
                    arr.append([speciesId, callTmin, callTmax, callFmin, callFmax, 0.8, 0.8, callTypeId])
                    callInserted = True
                if t2 < callTmin or f1 > callFmax or t1 > callTmax or t2 < callTmin:
                    arr.append([id, t1, t2, f1, f2, p1, p2, ct]) # exclude overlapped calls
        if not callInserted: arr.append([speciesId, callTmin, callTmax, callFmin, callFmax, 0.8, 0.8, callTypeId])
        print(f"BatCalls.Insert before{len(self.CallsNP)=}")
        self.CallsNP = numpy.array(arr, dtype='f')
        print(f"BatCalls.Insert after{len(self.CallsNP)=}")
        
    def FindFirstConsecutive(self):
        minT = 0
        if self.CallsNP is not None: 
            n = 0
            while n +1 < len(self.CallsNP) and self.CallsNP[n+1, 0] != self.CallsNP[n, 0] and (self.CallsNP[n+1, 1] - self.CallsNP[n, 1]) > 0.3: 
                n += 1 #more calls and not same species and too far apart
            if self.CallsNP.shape[0] > 0: 
                firstConsecutiveCallx = self.CallsNP[n, 1]
                if firstConsecutiveCallx > 0.01: minT =  firstConsecutiveCallx - 0.01
            print(f"FindFirstConsecutive {minT=}")                    
        return minT

    def GetSpeciesList(self):
        speciesList = []
        if self.CallsNP.shape[0] > 0: 
            idList = numpy.unique(self.CallsNP[:, 0])
            if self.SpeciesLanguage != "None":
                for id in idList:                 
                    sl = self.SpeciesLanguage;
                    if sl == "EnglishAbbrev": sl = "English" # otherwise not unique                    
                    speciesList.append(self.SpeciesNames.loc[id][sl])
        return speciesList
        
    def FindSpeciesMaxProb(self, id):
        condition = (self.CallsNP[:,0] == id)
        selected_rows = self.CallsNP[condition]
        c = numpy.argmax(selected_rows[:, 6]) #max probaliity
        minT = selected_rows[c, 1]
        print(f"FindFirstConsecutive {minT=}")                    
        return minT
