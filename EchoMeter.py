import xml.etree.ElementTree as Xml
import sys, os, folium, soundfile, polars, shutil, traceback, colorama
from Classifier import Classifier

class EchoMeter():
    """Deals with Wildlife Accoustics Echo Meter GPS based recordings"""
    def __init__(self, parentSelf):
        self.SpeciesNames = parentSelf.SpeciesNames
        self.FullSpeciesLanguage = parentSelf.FullSpeciesLanguage
        self.AbbrevSpeciesLanguage = parentSelf.AbbrevSpeciesLanguage
        self.CallTypes = parentSelf.CallTypes
        self.Status = parentSelf.Status
        sys.excepthook = self.notify_exception
        self.EmNameIdx = dict(zip(self.SpeciesNames["EchoMeter"], self.SpeciesNames.index)) 

    def LoadEchoMeterDir(self, echoMeterPath):
        self.echoMeterPath = echoMeterPath
        self.classify = Classifier()
        self.TE_DirPath = os.path.join(self.echoMeterPath, "TimeExpanded")
        gpsBatFilePath = os.path.join(self.echoMeterPath, "GpsBatCallFiles.csv")
        if os.path.exists(gpsBatFilePath):
            self.GpsFilesDF = polars.read_csv(gpsBatFilePath)
        else:
            self.GpsFilesDF = polars.DataFrame(schema=[("SessionName", polars.Utf8), ("Filename", polars.Utf8), ("DateTime", polars.Utf8), 
                ("Lat", polars.Float64), ("Long", polars.Float64), ("Class", polars.Utf8), ("Abbrev", polars.Utf8), ("Species", polars.Utf8) ]) # Utf8 = string
            if not os.path.exists(self.TE_DirPath):
                os.makedirs(self.TE_DirPath)
        print("LoadEchoMeterDir", echoMeterPath)
        basename = os.path.basename(echoMeterPath)
        if basename.startswith("Session_"):
            # single directory
            if basename not in self.GpsFilesDF["SessionName"].values:
                self.DecodeSession(echoMeterPath)
                self.GpsFilesDF.write_csv(os.path.join(self.echoMeterPath, 'GpsBatCallFiles.csv'))
        else:
            # multiple sub-directories
            self.LoadNewSessions()
        return self.GpsFilesDF 
            
    def LoadNewSessions(self):
        for session in os.listdir(self.echoMeterPath):
            print(f"LoadNewSessions {session=} {len(self.GpsFilesDF)=}")
            if session.startswith("Session_"):
                sessionExists = self.GpsFilesDF.select(polars.col("SessionName").is_in([session]).any()).item()
                print(f"LoadNewSessions {sessionExists=}")
                if sessionExists:
                    print(f"LoadNewSessions {session} already exists")
                else:
                    self.DecodeSession(os.path.join(self.echoMeterPath, session))
        self.GpsFilesDF.write_csv(os.path.join(self.echoMeterPath, 'GpsBatCallFiles.csv'))

    def SaveEchoMeterDir(self, GpsFilesDF=None):
        if GpsFilesDF is not None:
            self.GpsFilesDF.write_csv(os.path.join(self.echoMeterPath, 'GpsBatCallFiles.csv'))
            
    def SaveMap(self, Satellite=False, GpsFilesDF=None, separateRubbish=False):
        if GpsFilesDF is not None:
            self.GpsFilesDF = GpsFilesDF.drop_nulls()
            self.GpsFilesDF.write_csv(os.path.join(self.echoMeterPath, 'GpsBatCallFiles.csv'))      
        avgLat = self.GpsFilesDF.select(polars.col("Lat").mean()).item()
        avgLong = self.GpsFilesDF.select(polars.col("Long").mean()).item()
        map = folium.Map(location= [avgLat, avgLong] , zoom_start=17, width='100%', height='100%',
            tiles='https://tile.openstreetmap.org/{z}/{x}/{y}.png', attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')
        if Satellite:
            tile = folium.TileLayer( tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri', name = 'Esri Satellite', overlay = False, control = True).add_to(map)
        if separateRubbish:
            rubbishDirPath = self.echoMeterPath + "/rubbish"
            os.rename(self.TE_DirPath, rubbishDirPath)
            os.makedirs(self.TE_DirPath)
        for row in self.GpsFilesDF.iter_rows(named=True):
            if row["Species"] != 'NoID' and  row["Species"] != 'None':
                aText = f'"TimeExpanded/{row["Filename"]}"'
                pText = f'{row["Species"].replace(" ","_")}<br>{row["DateTime"].replace(" ","_")}<br><a href={aText}>Play sound 1/10</a>'
                popup = folium.Popup(pText, max_width=200)
                folium.Marker(location=[row['Lat'], row['Long']], popup=pText,
                    icon=folium.DivIcon(html=f"""<div style="font-family: courier new; font-weight: bold; color: blue">{row['Abbrev']}</div>""")).add_to(map)
                if separateRubbish: shutil.move(f'{os.path.join(rubbishDirPath, row["Filename"])}', f'{os.path.join(self.TE_DirPath, row["Filename"])}')
        mapPath = os.path.join(self.echoMeterPath, 'Bat Map.html')
        map.save(mapPath)
        return mapPath

    def SaveAudio(self, filepath, recording, sampleRate, factor = 0.1):
        loud = recording * 10
        soundfile.write(filepath, loud, round(sampleRate * factor)) 

    def DecodeSession(self, SessionDirPath, debug=False):		
        session = os.path.basename(SessionDirPath)
        r = 0
        print(f"DecodeXml {session=}")
        tree = Xml.parse(os.path.join(SessionDirPath, session + ".kml")) 
        root = tree.getroot()
        #if debug: print(f"DecodeXml {root.tag=}, {root.attrib=}")
        vEnd = root.tag.find('}') +1
        for c1 in root:
            tag1 = c1.tag[vEnd:]
            #if debug: print(f"   1 {tag1} = {c1.attrib}, {c1.text}")
            for c2 in c1:
                tag2 = c2.tag[vEnd:]
                if tag2 == 'Placemark':
                    abbrev = ''
                    #if debug: print(f"      2 {tag2} = {c2.attrib}, {c2.text}")
                    for c3 in c2:
                        tag3 = c3.tag[vEnd:]
                        if tag3 == 'LineString': # LineString for path walked
                            continue
                        if tag3 == 'name': #recording filename
                            recordingFile = c3.text + ".wav"
                        for c4 in c3:
                            tag4 = c4.tag[vEnd:]
                            if tag4 == 'coordinates':
                                latLong = c4.text.split(',')
                                lat = latLong[1]
                                long =latLong[0]
                            #if debug: print(f"            4 {tag4} = {c4.attrib}, {c4.text}")
                            for c5 in c4:
                                tag5 = c5.tag[vEnd:]
                                if tag4 == 'Data' and c4.attrib['name'] == 'Bat Type' and tag5 == 'value':
                                    i = self.EmNameIdx[c5.text]
                                    species = self.SpeciesNames[self.FullSpeciesLanguage][i]
                                    abbrev = self.SpeciesNames[self.AbbrevSpeciesLanguage][i]
                    aEnd = recordingFile.find('_') +1
                    a = recordingFile[aEnd:]
                    if len(a) >= 15 and abbrev != '':
                        audioPath = os.path.join(SessionDirPath, recordingFile)
                        if os.path.exists(audioPath):
                            r += 1
                            self.Status(f"Processing Google map file {session + ".kml"} recording {r}")
                            results = self.classify.File(audioPath)
                            if len(results) > 0: 
                                dateTime =  f"{a[6]+a[7]}/{a[4]+a[5]}/{a[2]+a[3]} {a[9]+a[10]}:{a[11]+a[12]}:{a[13]+a[14]}"
                                new_row = polars.DataFrame({ "SessionName": [session], "Filename": [recordingFile],  "DateTime": [dateTime], "Lat": [float(lat)], "Long": [float(long)], "Class": [results], "Abbrev": [abbrev], "Species": [species],})
                                self.GpsFilesDF.extend(new_row)
                                recording, sampleRate = soundfile.read(audioPath)
                                self.SaveAudio(os.path.join(self.TE_DirPath, recordingFile), recording, sampleRate, factor=0.1) 

    def notify_exception(self, type, value, tb):
        traceback_details = "\n".join(traceback.extract_tb(tb).format())
        msg = f"caller: {' '.join(sys.argv)}\n{type}: {value}\n{traceback_details}"
        print(colorama.Fore.RED + msg + colorama.Fore.RESET)
        self.Status("EXCEPTION see console", error=True)