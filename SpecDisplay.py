import dearpygui.dearpygui as dpg
import sys, os, subprocess
if sys.platform.startswith("win"): 
    import DearPyGui_DragAndDrop as DragAndDrop
    os.add_dll_directory(os.path.dirname(__file__) + "/ffmpeg") # allows for remote batch files
import numpy, soundfile, sounddevice, time, warnings, scipy, traceback, colorama
import pandas, re, multiprocessing, ctypes, math, torchaudio, torch, torchaudio_filters
import scipy.signal, scipy.io.wavfile # filter for ref calls
import utils
from Classifier import Classifier
from BatCalls import BatCalls
from EchoMeter import EchoMeter
from torchcodec.decoders import AudioDecoder
numpy.set_printoptions(precision=3, suppress=True)
numpy.set_printoptions(threshold=sys.maxsize)

MIN_FREQ_KHZ = 0; MAX_FREQ_KHZ = 125; STD_SAMPLING = 250000; MAX_PLAY_RATE = 40000; MAX_PLAY_SEC = 30; LOUDNESS = 10
NFFT = 512; RELATIVE_HOP_LENGTH = 0.5 # spectogram settings
PSD_WIDTH = 80;  SLIDER_W = 17; AMP_HT=80; SCROLL_HT=19; BUTTON_HT=19; STATUS_HT=24; SPACING=7; HEADER=30; COLOR_SCALE_W=55
ROW_PXL = 17 # table scrolling

class SpecDisplay(): 
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
        
    def __init__(self, parent, config, parentSelf, activeButtonCallback, showDisplay=True, showAmp=True):
        self.SpeciesNames = parentSelf.SpeciesNames
        self.SpeciesLanguage = parentSelf.SpeciesLanguage
        self.Status = parentSelf.Status
        self.classifyEnabled = True
        sys.excepthook = self.notify_exception
        self.FileTableRow = self.lastMousePlotPos = self.lastMousePos = self.soundLine = self.duration = self.dirIndex = self.FilesDF = self.PlayObject = None 
        self.activeButtonCallback = activeButtonCallback        
        self.minF = MIN_FREQ_KHZ; self.maxF = MAX_FREQ_KHZ
        self.colours = self.GenerateSpectrum()
        self.colormapRegistry = dpg.add_colormap_registry(label="Colormap Registry")
        self.colorMap = dpg.add_colormap(self.colours, qualitative=True, parent=self.colormapRegistry)
        YtickLabels = self.GenerateYtickLabels()        
        self.minT = config["minT"]; self.maxT = config["maxT"]; self.dir = config["dir"]; self.file = config["file"]        
        self.EditMode = config["EditMode"]
        self.timeStep = self.Range = float(config["Range"])
        self.minF = float(config["minF"]) 
        self.maxF = float(config["maxF"]) 
        self.HighPassFreq = self.minF * 1000                     
        self.LowPassFreq = self.maxF * 1000 
        self.soundProgressBar = self.heatSeries = self.ampSeries = self.psdSeries = self.ZoomStart = self.LabelStartPlot = None
        self.maxPercent = 100; self.minPercent = 0
        self.calls = BatCalls(parentSelf)
        
        specHeight = config["height"] * 0.8 - (AMP_HT + SCROLL_HT + 2*BUTTON_HT + STATUS_HT / 2+ HEADER ) * config["scale"] - SPACING *7
        with dpg.group(horizontal=True, show=showDisplay, height=specHeight):
            self.topGroup = dpg.last_item()
            self.MinSlider = dpg.add_slider_int(width=SLIDER_W * config["scale"], height = -1, callback=self.MinSlider_callback, default_value=self.minPercent, vertical=True, clamped=True, min_value=0, max_value=100)
            self.MaxSlider  = dpg.add_slider_int(width=SLIDER_W * config["scale"], height = -1, callback=self.MaxSlider_callback, default_value=self.maxPercent, vertical=True, clamped=True, min_value=0, max_value=100)
            self.colormap_scale = self.colormap_scale1 = dpg.add_colormap_scale(width=COLOR_SCALE_W * config["scale"], height = -1, colormap=self.colorMap, format="%.2f")
            with dpg.plot(height=-1, label="Power", no_menus=False, width=PSD_WIDTH * config["scale"], crosshairs=True):
                self.psdPlot = dpg.last_item()
                self.psdXaxis = dpg.add_plot_axis(dpg.mvXAxis, label="", no_gridlines=True)
                self.psdYaxis = dpg.add_plot_axis(dpg.mvYAxis, no_tick_labels=True, no_gridlines=False, foreground_grid=True)
                dpg.set_axis_ticks(self.psdYaxis, YtickLabels)
            with dpg.plot(label="Spectogram", height=-1, width=-1, crosshairs=True, pan_button=-1, pan_mod=2, box_select_button=dpg.mvMouseButton_Left, no_menus=True):
                self.specPlot = dpg.last_item()
                self.specXaxis = dpg.add_plot_axis(dpg.mvXAxis, label="Time seconds", tick_format="%.3f", foreground_grid=True)
                self.specYaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Frequency kHz", foreground_grid=True )
                dpg.set_axis_ticks(self.specYaxis, YtickLabels)
        dpg.bind_colormap(self.specPlot, self.colorMap)            

        with dpg.group(horizontal=True, show=showDisplay):
            self.bottomGroup = dpg.last_item()
            self.activeDisplayText = dpg.add_button(callback=self.ActiveDisplay_click, height=SCROLL_HT * config["scale"] *2 , width=(PSD_WIDTH + SLIDER_W *2 + COLOR_SCALE_W + SPACING*3) * config["scale"])
            with dpg.group(horizontal=False):
                with dpg.plot(no_title=True, show=showAmp, height=AMP_HT * config["scale"], width=-1):
                    self.ampPlot = dpg.last_item()
                    self.ampXaxis = dpg.add_plot_axis(dpg.mvXAxis, tick_format="%.3f", foreground_grid=True)
                    self.ampYaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", no_gridlines=True, foreground_grid=True)
                with dpg.group(horizontal=True, height=SCROLL_HT * config["scale"]):
                    dpg.add_text("Scroll")
                    self.ScrollBar = dpg.add_slider_float(width=-1, callback=self.ScrollBar_callback, default_value=self.minT, vertical=False, clamped=True, min_value=0, max_value=1)
                self.sliderWidth = config["width"] - 40 - PSD_WIDTH
                with dpg.group(horizontal=True, height=SCROLL_HT * config["scale"]):
                    PlaySoundbutton = dpg.add_button(label="Play Sound", callback=self.PlaySound)
                    self.PlaySpeedCombo = dpg.add_combo(label="Speed", items=("1", "1/2", "1/5", "1/10", "1/20"), width=66*config["scale"], default_value=f"1/10")
                    saveSoundbutton = dpg.add_button(label="Save Sound",  callback=self.saveSound_click)
                    self.showSpeciesCombo = dpg.add_combo(label="Find Species", show=False, width=180 * config["scale"], callback=self.ShowSpeciesCombo_changed)
                    self.ClassifyLabel = dpg.add_text(color= (0, 200, 0, 255))                
        with dpg.theme() as lBlueButton_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 175, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 0, 115, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 0, 40, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (185, 185 , 185), category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.greenText_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0, 255), category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.defaultText_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (180, 180, 180, 255), category=dpg.mvThemeCat_Core)
        with dpg.theme() as orangeText_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255,165,0, 255), category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.line_theme:
           with dpg.theme_component(dpg.mvLineSeries):
                  dpg.add_theme_color(dpg.mvPlotCol_Line, (200, 200, 200, 255), category=dpg.mvThemeCat_Plots)
        with dpg.theme() as minSlider_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, self.colours[1], category=dpg.mvThemeCat_Core)
        with dpg.theme() as maxSlider_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, self.colours[255], category=dpg.mvThemeCat_Core)                
        dpg.bind_item_theme(PlaySoundbutton, lBlueButton_theme)
        dpg.bind_item_theme(saveSoundbutton, lBlueButton_theme)
        dpg.bind_item_theme(self.showSpeciesCombo, orangeText_theme) 
        dpg.bind_item_theme(self.MinSlider, minSlider_theme)
        dpg.bind_item_theme(self.MaxSlider, maxSlider_theme)
        self.classify = Classifier()

    def LoadClassifiedFile(self, filepath, rememberDir=True):
        titleExtra = ""
        dir = os.path.dirname(filepath); file = os.path.basename(filepath)
        if rememberDir: 
            if self.dir != dir or self.dirIndex is None:
                self.RememberDirectory(dir, filepath)
            titleExtra = f"file {self.dirIndex +1} of {len(self.dirFiles)}"
        self.dir = dir; self.file = file
        if self.classifyEnabled:            
            callsCsvPath = os.path.join(dir,"ann", file+".csv")
            if os.path.isfile(callsCsvPath):
                summary = self.calls.fromCSV(callsCsvPath)
                self.SetClassifyLabel(summary)                
                speciesComboList = self.calls.GetSpeciesList()
                if len(speciesComboList) > 1:
                    dpg.configure_item(self.showSpeciesCombo,  items=speciesComboList)
                    dpg.configure_item(self.showSpeciesCombo, show=True)
                else:
                    dpg.configure_item(self.showSpeciesCombo, show=False)            
            else:
                dpg.set_value(self.ClassifyLabel, "No bat calls found")
                dpg.configure_item(self.ClassifyLabel, color=(200, 0, 0, 255))
        self.LoadFile(filepath, titleExtra)  

    def GenerateSpectrum(self):
        colours = []
        for x in range(0, 256):
            r = 0; g = 0; b = 0
            if x > 0 and x <= 60: b = int(x * 3.3 + 57)
            if x > 60 and x < 120: b = int(255 - (x -60)* 3)
            if x > 40 and x <= 140: g = int((x -40)* 2.1)
            if x > 140 and x < 240: g = int(255 - (x -140)* 2.1)
            if x > 120: r = int((x-120) * 1.9)
            #print(f"GenerateSpectrum {x=} {r=} {g=} {b=}")
            colours.append([r, g, b])
        return colours
        
    def GenerateYtickLabels(self):
        x = 0
        YtickLabels = list([])
        for y in range(25):
            f = y * 5
            if f == round(f/10)*10: tick = (f"{f:.0f}", float(f))
            else: tick = ("", float(f))                
            YtickLabels.append(tick)
        return tuple(YtickLabels)
        
    def RememberDirectory(self, dir, f):
        print(f"RememberDirectory {dir=} {f=}")
        # will include files with no classified calls
        self.dirFiles = utils.ListAudioFiles(dir)
        i = 0
        for filepath in self.dirFiles:
            if f == filepath:
                self.dirIndex = i; 
                break
            i += 1
    
    def ActiveDisplay_click(self):
        self.activeButtonCallback()
    
    def ShowActiveDisplay(self, active=True):
        if active: 
            dpg.set_item_label(self.activeDisplayText, "Active Display for:\nfile drops and arrow keys")
            dpg.bind_item_theme(self.activeDisplayText, self.greenText_theme) 
        else: 
            dpg.set_item_label(self.activeDisplayText, "Press to activate")
            dpg.bind_item_theme(self.activeDisplayText, self.defaultText_theme) 
        
    def ScrollBar_callback(self, sender, app_data, user_data):
        if app_data + self.Range > self.duration: self.maxT = self.duration; self.minT = self.duration - self.Range
        else: self.minT = app_data; self.maxT = app_data + self.Range
        self.DisplaySpectogram()    
    
    def MinSlider_callback(self, sender, app_data, user_data):
        print(f"MinSlider_callback {app_data=}")
        self.minPercent = app_data
        self.DisplaySpectogram(UpdateMin= False, sound=False)

    def MaxSlider_callback(self, sender, app_data, user_data):
        print(f"MaxSlider_callback {app_data=}")
        self.maxPercent = app_data
        self.DisplaySpectogram(UpdateMin= False, sound=False)
        
    def ShowSpeciesCombo_changed(self, sender, app_data, user_data):
        sl = self.SpeciesLanguage
        if sl == "EnglishAbbrev": sl = "English"
        id = self.SpeciesNames.index[self.SpeciesNames[sl]==app_data].tolist()
        print(f"ShowSpeciesCombo_changed {app_data=} {s=}")
        t1 = self.calls.FindSpeciesMaxProb(id)
        if t1 > self.Range / 2: 
            if t1 + self.Range / 2 < self.duration: 
                self.minT =  t1 - self.Range / 2; self.maxT =  t1 + self.Range / 2
            else:
                self.maxT = self.duration; self.minT =  t1 - self.Range
        else: 
            self.minT = 0; self.maxT = t1 + self.Range
        self.DisplaySpectogram()
        dpg.set_value(self.ScrollBar, self.minT) 

    def LoadFile(self, filepath, titleExtra=""):
        self.Status("")
        filename = os.path.basename(os.path.splitext(filepath)[0])
        self.decoder = AudioDecoder(filepath)
        self.sample_rate = self.decoder.metadata.sample_rate
        if filename.endswith("TE"):
            self.timeExpand = True            
            self.sample_rate *= 10
            c = f"Time expanded file increasing sampling rate to {self.sample_rate}"
            print(f"LoadFile {c}")
            self.Status(c)
        else:  self.timeExpand = False
        if self.decoder.metadata.begin_stream_seconds_from_header is None: self.duration = self.decoder.metadata.duration_seconds_from_header
        else: self.duration = self.decoder.metadata.duration_seconds_from_header - self.decoder.metadata.begin_stream_seconds_from_header
        print(f"LoadFile {filepath=}  {self.duration=} {titleExtra=}")
        userPath = os.path.expanduser("~")
        if userPath.lower() in filepath.lower():
            dpg.set_item_label(self.specPlot, f"Spectrogram of {filepath[len(userPath)+1:]} {utils.FileDate(filename)} {titleExtra}")
        else: dpg.set_item_label(self.specPlot, f"Spectrogram of {filepath} {utils.FileDate(filename)} {titleExtra}")
        dpg.configure_item(self.ScrollBar, max_value=self.duration - self.Range) 
        dpg.configure_item(self.ScrollBar, format=f"%.03f of {self.duration:.3f} secs")         
        dpg.set_item_label(self.ScrollBar, f" of {self.duration}") 
        dpg.set_value(self.ScrollBar, self.minT) 
        grabSize = self.sliderWidth * self.Range / self.duration
        with dpg.theme() as slider_theme:
            with dpg.theme_component(dpg.mvSliderFloat):
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, grabSize, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme(self.ScrollBar, slider_theme)
        self.dir = os.path.dirname(filepath)
        self.file = os.path.basename(filepath)
        self.minT = self.calls.FindFirstConsecutive()
        dpg.set_value(self.ScrollBar, self.minT) 
        if self.minT + self.Range > self.duration: 
            self.maxT = self.duration
            if self.minT > self.Range: self.minT = self.duration - self.Range
            else: self.minT  = 0
        else: self.maxT = self.minT + self.Range
        dpg.set_value(self.ScrollBar, self.minT)
        self.DisplaySpectogram()
        if self.sample_rate < 45000:
            self.Status(f"Sample rate = {self.sample_rate / 1000:.1f}kHz FILE NOT ULTRASONIC")
        
    def LoadFileSegment(self):
        print(f"LoadFileSegment {self.minT=} {self.maxT=} {self.minF=} {self.maxF=} {self.HighPassFreq=}")
        if self.timeExpand: waveformTensor = self.decoder.get_samples_played_in_range(start_seconds=self.minT*10, stop_seconds=self.maxT*10)
        else: waveformTensor = self.decoder.get_samples_played_in_range(start_seconds=self.minT, stop_seconds=self.maxT)
        waveformTensor = waveformTensor.data
        if self.sample_rate <= STD_SAMPLING: nfft=NFFT
        elif self.sample_rate > STD_SAMPLING: 
            nfft = int(NFFT * self.sample_rate / STD_SAMPLING) # allow for extra frequencies
            print(f"LoadFileSegment {self.sample_rate=} > {STD_SAMPLING} reducing {NFFT=} to {nfft=}")        
        specTransform = torchaudio.transforms.Spectrogram(n_fft=nfft, hop_length=int(nfft*RELATIVE_HOP_LENGTH), power=1, window_fn=torch.blackman_window)#power: 1=magnitude, 2=power
        spectrogram = specTransform(waveformTensor) # [Channels, Frequency Bins ,Time Steps]
        #print(f"LoadFileSegment specTransform {waveformTensor.shape=} = {spectrogram.shape=}")    
        if self.sample_rate > STD_SAMPLING: 
            n =  NFFT // 2 +1
            #print(f"LoadFileSegment top {spectrogram.shape[1] - n} frequencies cut off")
            spectrogram = spectrogram[:, :n, :]# cut off higher frequencies
        elif self.sample_rate < STD_SAMPLING:
            #n = NFFT // 2 +1 - spectrogram.shape[1]
            n = int(spectrogram.shape[1] / self.sample_rate * STD_SAMPLING - spectrogram.shape[1])
            before = spectrogram.shape[1]
            spectrogram = torch.nn.functional.pad(input=spectrogram, pad=(0,0,0,n,0,0), mode='constant', value=0) # add padding of high frequencies
            #print(f"LoadFileSegment pad higher frequencies {n=} {before=} -> {spectrogram.shape[1]=}")
        recordingLength = waveformTensor.shape[1]/self.sample_rate
        #print(f"LoadFileSegment before  {spectrogram.shape=} / {self.sample_rate=} = {recordingLength=}, {self.Range=}")
        if recordingLength < self.Range:
            n = int(spectrogram.shape[2] * self.Range / recordingLength - spectrogram.shape[2])
            spectrogram = torch.nn.functional.pad(input=spectrogram, pad=(0,n,0,0,0,0), mode='constant', value=0) # add padding of time
            self.maxT = self.minT + self.Range
        #print(f"torchaudio {waveformTensor.shape=}, {self.sample_rate=}")
        self.freqBins = spectrogram.shape[1]; self.timeSteps= spectrogram.shape[2]
        #print(f"LoadFileSegment {waveformTensor.nbytes=}, {spectrogram.nbytes=}, channels={spectrogram.shape[0]}, {self.freqBins=}, {self.timeSteps=}")
        if self.HighPassFreq > 0:
            if self.sample_rate > self.HighPassFreq * 2:
                highPassFilter = torchaudio_filters.HighPass(self.HighPassFreq, self.sample_rate) 
                waveformTensor = highPassFilter(waveformTensor)
            else:
                self.Status(f"{self.sample_rate=} TOO LOW FOR {self.HighPassFreq=}", error=True)
        
        if self.LowPassFreq < MAX_FREQ_KHZ:
            lowPassFilter = torchaudio_filters.Lowass(self.LowPassFreq, self.sample_rate) 
            waveformTensor = lowPassFilter(waveformTensor)

        if self.maxF < MAX_FREQ_KHZ:
            removeHighBins = int(self.freqBins / MAX_FREQ_KHZ * (MAX_FREQ_KHZ - self.maxF))
            spectrogram = spectrogram[:, :-removeHighBins, :]
            self.freqBins = spectrogram.shape[1]

        if self.minF > 0.0:
            removeLowBins = int(self.freqBins / self.maxF * self.minF)
            spectrogram = spectrogram[:, removeLowBins:, :]
            self.freqBins = spectrogram.shape[1]
        
        if spectrogram.shape[0] > 1:
            spectrogram = torch.mean(spectrogram, dim=0).unsqueeze(0) # stereo to mono
            print(f"LoadFileSegment {spectrogram.shape[0]} channels to mono")
        return spectrogram, waveformTensor[0].numpy()
    
    def RectOnSpec(self, startPoint, endPoint):
        npSpec = numpy.copy(self.npSpec)
        freqPerPixel = (self.maxF - self.minF) /self.freqBins #1
        timePerPixel = (self.maxT - self.minT) / self.timeSteps #2
        x1 = (startPoint[0] - self.minT)/ timePerPixel; x2 = (endPoint[0] - self.minT)/ timePerPixel
        y1 = (startPoint[1] - self.minF)/ freqPerPixel; y2 = (endPoint[1] - self.minF)/ freqPerPixel
        minY = int(min(y1, y2)) -1; minX = int(min(x1, x2)) -1; maxY = int(max(y1, y2)) +1; maxX = int(max(x1, x2)) +1
        npSpec[minY:maxY, minX] = self.maxA; npSpec[minY:maxY, maxX] = self.maxA;npSpec[minY, minX:maxX] = self.maxA; npSpec[maxY, minX:maxX] = self.maxA
        values = numpy.flipud(npSpec).flatten().tolist()
        if self.heatSeries is not None: dpg.delete_item(self.heatSeries); self.heatSeries = None
        self.heatSeries = dpg.add_heat_series(values, rows=self.specRows, cols=self.specCols, parent=self.specYaxis, 
            format="", scale_min=self.minA, scale_max=self.maxA, bounds_min=[self.minT,self.minF], bounds_max=[self.maxT,self.maxF]) 
        dpg.configure_item(self.colormap_scale, min_scale=self.minA)
        dpg.configure_item(self.colormap_scale, max_scale=self.maxA)
        dpg.fit_axis_data(self.specXaxis) # cancel any zoom ??
        dpg.fit_axis_data(self.specYaxis)        
        
    def DisplaySpectogram(self, UpdateMin= True, sound = True):
        if not dpg.get_item_configuration(self.topGroup)['show']: return
        if self.lastMousePlotPos is not None: 
            self.lastMousePos = self.lastMousePlotPos = None
        #print(f"DisplaySpectogram {self.minT=:.3f}, {self.maxT=:.3f}, {self.minF=}, {self.maxF=}") 
        self.zoomed = False
        spectrogram, self.Recording = self.LoadFileSegment()
        self.ZoomRecording = None
        self.npSpec = spectrogram[0].numpy()
        if UpdateMin:
            hist, bin_edges = numpy.histogram(self.npSpec, 100)
            self.minPercent = int(hist.argmax()+1)
            dpg.set_value(self.MinSlider, self.minPercent)
        #print(f"DisplaySpectogram {self.npSpec.shape=}")
        self.minA = self.npSpec.min(); self.maxA= self.npSpec.max()
        pRange = (self.maxA - self.minA) / 100
        self.minA += pRange * self.minPercent
        self.maxA = self.minA + pRange * self.maxPercent
        #print(f"DisplaySpectogram {self.minA=}, {self.maxA=}")
        values = numpy.flipud(self.npSpec).flatten().tolist()
        self.specRows = self.npSpec.shape[0]; self.specCols = self.npSpec.shape[1] ##################
        if self.heatSeries is not None: dpg.delete_item(self.heatSeries); self.heatSeries = None
        self.heatSeries = dpg.add_heat_series(values, rows=self.specRows, cols=self.specCols, parent=self.specYaxis, 
            format="", scale_min=self.minA, scale_max=self.maxA, bounds_min=[self.minT,self.minF], bounds_max=[self.maxT,self.maxF]) 
        dpg.configure_item(self.colormap_scale, min_scale=self.minA)
        dpg.configure_item(self.colormap_scale, max_scale=self.maxA)
        dpg.fit_axis_data(self.specXaxis) # cancel any zoom
        dpg.fit_axis_data(self.specYaxis)
        
        t = numpy.arange(self.Recording.shape[0]) / self.sample_rate + self.minT
        if self.ampSeries is not None: dpg.delete_item(self.ampSeries) 
        if not self.Recording.flags['C_CONTIGUOUS']: self.Recording = self.Recording.copy(order='C')
        self.ampSeries = dpg.add_line_series(t, self.Recording, parent=self.ampYaxis)
        dpg.bind_item_theme(self.ampSeries, self.line_theme)
        dpg.set_axis_limits(self.ampXaxis, self.minT, self.maxT)
        dpg.set_axis_limits(self.ampYaxis, self.Recording.min(), self.Recording.max())
        
        psd_transform = torchaudio.transforms.PSD()
        self.npPsd = psd_transform(spectrogram).numpy()
        minPsd =  self.npPsd.min(); maxPsd = self.npPsd.max()
        if self.psdSeries is not None: dpg.delete_item(self.psdSeries) 
        powerF = numpy.arange(self.minF, self.maxF,  (self.maxF - self.minF)/ (self.freqBins-1))
        self.psdSeries = dpg.add_line_series(self.npPsd, powerF, parent=self.psdYaxis)
        dpg.set_axis_limits(self.psdYaxis, self.minF, self.maxF) 
        dpg.set_axis_limits(self.psdXaxis, minPsd, maxPsd)
        
        peak = numpy.argmax(self.npPsd) / (self.freqBins-1) * (self.maxF - self.minF) + self.minF
        dpg.set_item_label(self.psdXaxis, f"Peak {peak:.1f} kHz")
        
        dpg.bind_item_theme(self.psdSeries, self.line_theme)
        self.species, exception = self.calls.DisplayAnnotations(self.specPlot, self.minT, self.maxT)
        if len(exception) > 0: self.Status(exception, error=True)

        if sound: self.PlaySound(cursor=None)
    
    def Range_changed(self, range):
        oldRange = self.Range
        self.Range = range
        diff = self.Range - oldRange
        if  self.duration is not None:
            print(f"SpecDisplay Range_changed {self.Range=}")
            self.maxT += diff
            grabSize = self.sliderWidth * self.Range / self.duration
            with dpg.theme() as slider_theme:
                with dpg.theme_component(dpg.mvSliderFloat):
                    dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, grabSize, category=dpg.mvThemeCat_Core)
            dpg.bind_item_theme(self.ScrollBar, slider_theme)        
            self.DisplaySpectogram(UpdateMin= False, sound = False)    

    ############################################################
    
    def PlaySound(self, cursor=True):
        devices = sounddevice.query_devices()
        if len(devices) == 0:
            print("NO SOUND DEVICES")
            return
        sounddevice.stop()
        speed = eval(dpg.get_value(self.PlaySpeedCombo))
        if self.ZoomRecording is None: Recording = self.Recording
        else: Recording = self.ZoomRecording
        replayRate = self.sample_rate * speed
        duration = len(Recording) / replayRate
        maxLength = round(self.sample_rate * MAX_PLAY_SEC * speed)
        
        if replayRate > MAX_PLAY_RATE:
            # too fast to play
            downscale_factor = math.ceil(replayRate / MAX_PLAY_RATE)
            Recording = self.DownSample(Recording, downscale_factor)
            replayRate = int(replayRate / downscale_factor)
            
        print(f"SpecDisplay PlaySound {cursor=} {len(Recording)=}, {self.sample_rate=}, {speed=} {replayRate=}, {duration=:.1f}")
        if duration > MAX_PLAY_SEC:
            Recording = Recording[:maxLength] * LOUDNESS
            duration = MAX_PLAY_SEC
        else:   
            Recording *= LOUDNESS
            
        if duration > 1.0 and cursor is not None:
            self.PlaySoundAndProgress(Recording, replayRate, speed)
        else:
            sounddevice.play(Recording, replayRate)

    def saveSound_click(self):
        file,_ = os.path.splitext(self.file)
        if dpg.get_value(self.PlaySpeedCombo) == "1/10":
            filepath = os.path.join(self.dir, file + f"_{round(self.minT*1000)}ms_{self.species}_TE.wav")
            loud = self.Recording * 10
            soundfile.write(filepath, loud, round(self.sample_rate / 10)) 
            self.Status(f"Time expanded audio saved as '{filepath}'") 
        else:
            file,_ = os.path.splitext(self.file)
            filepath = os.path.join(self.dir, file + f"_{round(self.minT*1000)}ms_{self.species}.wav")
            soundfile.write(filepath, self.Recording, self.sample_rate) 
            self.Status(f"Normal speed audio saved as '{filepath}'") 
        
    def DownSample(self, arr, downscale_factor):
        total_elements = arr.shape[0]
        odd_elements = total_elements % downscale_factor
        #remove odd elements so no numpy reshaping error
        if odd_elements > 0: arr = arr[:-odd_elements]
        print(f"DownSample {total_elements=} after {arr.shape=} {odd_elements=}")
        reshaped_arr = arr.reshape(-1, downscale_factor)
        # Downsample the array by taking the mean of each block
        downsampled_arr = reshaped_arr.mean(axis=1)
        print(f"DownSample {reshaped_arr.shape=} after {downsampled_arr.shape=}")
        return downsampled_arr
        
    def PlaySoundAndProgress(self, recording, SampleRate, speed):
        print(f"PlaySoundAndProgress {len(recording)=} {SampleRate=} {speed=}")
        if self.soundLine is not None:
            self.soundLine = None
            self.SoundProcess.terminate()
            try:
                dpg.delete_item(main.soundLine) 
            except Exception as error:
                print(colorama.Fore.RED + f"PlaySoundAndProgress An exception occurred: {error}" + colorama.Fore.RESET)
        
        temp = os.path.join(os.getcwd(), "Resources", "temp.wav") # needs full path
        soundfile.write(temp, recording, int(SampleRate)) 
            
        if sys.platform.startswith("win"):        
            self.SoundProcess = subprocess.Popen(['./wavplayer/sounder.exe', temp] )  
        elif sys.platform.startswith("linux"):
            self.SoundProcess = subprocess.Popen(['./wavplayer/wavplay', temp] )  
        else:
            raise ImportError("wavplay doesn't support this system")
            
        rect = dpg.get_item_rect_size(self.specPlot)
        self.xLim = dpg.get_axis_limits(self.specXaxis)
        self.yLim = dpg.get_axis_limits(self.specYaxis)
        self.speed = speed
        self.SoundTime = self.xLim[0]
        self.SoundFactor = 1/ speed
        self.soundLine = dpg.add_plot_annotation(parent=self.specPlot, label="^", default_value=(self.SoundTime, self.yLim[1]), offset=(0, rect[1]),color=[255, 255, 255, 255])
        self.SoundStartTime = float(time.perf_counter()) + 0.2 # fudge factor
        
    def UpdateSoundLine(self, plot):
        if self.soundLine is not None:
            try:
                dpg.delete_item(self.soundLine) 
            except Exception as error:
                print(colorama.Fore.RED + f"dearpygui An exception occurred: {error}" + colorama.Fore.RESET)
            self.SoundTime = self.xLim[0] + (time.perf_counter() - self.SoundStartTime) * self.speed
            if self.SoundTime > self.xLim[1]:
                self.soundLine = None
                print(f"UpdateSoundLine {time.perf_counter() - self.SoundStartTime}")
            else:
                rect = dpg.get_item_rect_size(plot)
                self.soundLine = dpg.add_plot_annotation(parent=plot, label="^", default_value=(self.SoundTime, self.yLim[1]), offset=(0, rect[1]), color=[255, 255, 255, 255])
    
    def SetClassifyLabel(self, result):
        if len(result) > 0:
            dpg.set_value(self.ClassifyLabel, result)
            if result == "No summary file":
                dpg.configure_item(self.ClassifyLabel, color=(200, 0, 0, 255))
            else:
                dpg.configure_item(self.ClassifyLabel, color=(0, 200, 0, 255))
        else:
            dpg.set_value(self.ClassifyLabel, "No bat calls found")
            dpg.configure_item(self.ClassifyLabel, color=(200, 0, 0, 255))
            
    def notify_exception(self, type, value, tb):
        traceback_details = "\n".join(traceback.extract_tb(tb).format())
        msg = f"caller: {' '.join(sys.argv)}\n{type}: {value}\n{traceback_details}"
        print(colorama.Fore.RED + msg + colorama.Fore.RESET)
        self.Status("EXCEPTION see console", error=True)