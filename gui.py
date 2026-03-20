import dearpygui.dearpygui as dpg
import sys, os, subprocess, utils
if sys.platform.startswith("win"): 
    import DearPyGui_DragAndDrop as DragAndDrop
    os.add_dll_directory(os.path.join(os.path.dirname(__file__), "ffmpeg")) # allows for remote batch files
import numpy, soundfile, sounddevice, time, warnings, json, scipy, wakepy, json, colorama
import pandas, re, multiprocessing, ctypes, math, torchaudio, torch, torchaudio_filters, traceback, webbrowser, chime
import scipy.signal, scipy.io.wavfile # filter for ref calls
from Classifier import Classify
from EchoMeter import EchoMeter
from torchcodec.decoders import AudioDecoder
from screeninfo import get_monitors
from SpecDisplay import SpecDisplay
from FileDialog import FileDialog

PSD_WIDTH = 80;  SLIDER_W = 17; AMP_HT=80; SCROLL_HT=19; BUTTON_HT=19; STATUS_HT=24; SPACING=7; HEADER=30; COLOR_SCALE_W=55
CONFIG_FILE = "gui_Config.json"; EXAMPLE_FILE = os.path.join("Resources", "bats", "NoctuleFeedingBuzz.wav")
DISPLAY_ROWS=7; ROW_PXL = 17 # table scrolling
TITLE = "Bat Detect GUI"
    
class MainWindow():            
    def __init__(self):
        self.SpeciesNames = pandas.read_csv(os.path.join("Resources", "SpeciesNames.csv"))
        self.lastMousePos = self.ZoomStart = self.LabelStartPlot = self.StatusLabel = self.AssignSpeciesID = self.AssignCallTypeID = None
        self.lastRow = self.FileTableRow = self.FilesDF = self.SoundProcess = self.soundLine = self.lastMousePlotPos = None
        self.MultiFile = config["MultiFile"]
        print(f"MainWindow ___init__ {torch.cuda.is_available()=}")       

        with dpg.window(label=TITLE.replace(" ", ""), width=-1, height=-1, pos=(0, 0), tag=TITLE.replace(" ", "")):
            self.mainWindow = dpg.last_item()
            self.EditMode = config["EditMode"]; 
            self.SpeciesLanguage = config["SpeciesLanguage"] 
            self.CallTypes = ("Echolocation", "Social", "Feeding")
            self.Range = float(config["Range"]);
            self.SpecDisplay2 = SpecDisplay(self.mainWindow, config, parentSelf=self, activeButtonCallback=self.activeButton2, showDisplay=False, showAmp=False)
            self.ActiveDisplay = self.SpecDisplay1 = SpecDisplay(self.mainWindow, config, parentSelf=self, activeButtonCallback=self.activeButton1)
            self.SpecDisplay1.SpecBlankKfreq = self.SpecDisplay2.SpecBlankKfreq = float(config["SpecBlankKfreq"])
            self.SpecDisplay1.HighPassFreq = self.SpecDisplay2.HighPassFreq = self.SpecDisplay1.SpecBlankKfreq*1000                     

            with dpg.group(horizontal=True, height=BUTTON_HT * config["scale"]):               
                self.saveMapButton = dpg.add_button(label="Save Map", show=False, height=-1, callback=self.SaveMap_click)
                SelectFileButton = dpg.add_button(label="Open Dir/File ...", height=-1, callback=self.FileDialog_Show)
                self.RangeCombo = dpg.add_combo(label="Range", items=("0.25s", "0.5s", "1.0s", "2.0s", "5.0s", "10s", "15s"), width=60*config["scale"], default_value=f"{self.SpecDisplay1.Range}s", callback=self.RangeListbox_changed)
                self.FilterCombo = dpg.add_combo(label="Filter", items=("> 0 kHz", "> 5 kHz", "> 10 kHz", "> 15 kHz", "> 20 kHz", "> 25 kHz", "> 30 kHz", "> 35 kHz", "> 40 kHz", "> 50 kHz", "> 60 kHz"), width=95*config["scale"], default_value=f"> {self.SpecDisplay1.SpecBlankKfreq} kHz", callback=self.FilterListBox_changed)
                self.SpeciesLanguageCombo = dpg.add_combo(label="Species Language", items=("Latin","LatinAbbrev", "English", "EnglishAbbrev", "None"), width=115*config["scale"], default_value=self.SpeciesLanguage, callback=self.SpeciesLanguageCombo_changed)
                self.EditCombo = dpg.add_combo(label="Edit", items=("None", "Source", "Species Ref", "Train"), width=100*config["scale"], default_value=self.EditMode, callback=self.EditModeListbox_changed)
                self.AssignSpeciesCombo = dpg.add_combo(label="Assign Species", width=180 * config["scale"], callback=self.AssignSpeciesCombo_changed)
                self.AssignCallTypeID = 0
                self.AssignCallTypeCombo = dpg.add_combo(label="Call Type", items=self.CallTypes, width=110*config["scale"], default_value=self.CallTypes[self.AssignCallTypeID], callback=self.AssignCallTypeCombo_changed)
                
                self.HelpButton = dpg.add_button(label="Help", width=60*config["scale"], callback=self.HelpButton_pressed)                     
            with dpg.table(policy=dpg.mvTable_SizingFixedFit, scrollY=True, height=config["height"] * 0.2):
                self.FileTable = dpg.last_item()
                dpg.add_table_column(label="Filename")
                dpg.add_table_column(label="Bat Calls")

            self.StatusLabel = dpg.add_button(width=-1, height=STATUS_HT * config["scale"])
            sys.excepthook = self.notify_exception
            
            self.FileDialog = FileDialog(self.FileDialog_Finished)
            
        BACKGROUND_COLOUR = (0,0,0)
        with dpg.theme() as self.align_right:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 1.00, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, BACKGROUND_COLOUR, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, BACKGROUND_COLOUR, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, BACKGROUND_COLOUR, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255 , 255), category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.red_align_right:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 1.00, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, BACKGROUND_COLOUR, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, BACKGROUND_COLOUR, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, BACKGROUND_COLOUR, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0 ,0), category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme(self.StatusLabel, self.align_right)
        
        with dpg.font_registry():
            default_font = dpg.add_font("./Resources/Swansea-q3pd.ttf", config["font"])
            bold_font = dpg.add_font("./Resources/SwanseaBold-D0ox.ttf", int(config["font"] * 1.5))
        if config["scale"] < 0.8 or config["scale"] > 1.2: 
            print(f"MainWindow default_font altered to Swansea-q3pd.ttf {config['font']=} {config['scale']=}")
            dpg.bind_font(default_font)
        dpg.bind_item_font(self.StatusLabel, bold_font)
        
        with dpg.theme() as greenButton_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 100, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 125, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 150, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10, category=dpg.mvThemeCat_Core)
        with dpg.theme() as magentaButton_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (119, 0, 119, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (139, 0, 139, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (149, 0, 149, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10, category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.magentaText_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 255, 255), category=dpg.mvThemeCat_Core)
        with dpg.theme() as orangeText_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255,165,0, 255), category=dpg.mvThemeCat_Core)
        with dpg.theme() as blueProgress_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (0, 0, 100, 255), category=dpg.mvThemeCat_Core)
        with dpg.theme() as table_theme:
            with dpg.theme_component(dpg.mvTable):
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.line_theme:
           with dpg.theme_component(dpg.mvLineSeries):
                  dpg.add_theme_color(dpg.mvPlotCol_Line, (200, 200, 200, 255), category=dpg.mvThemeCat_Plots)                
        dpg.bind_item_theme(SelectFileButton, greenButton_theme)
        dpg.bind_item_theme(self.FileTable, table_theme)
        dpg.bind_item_theme(self.saveMapButton, magentaButton_theme)
        dpg.bind_item_theme(self.EditCombo, self.magentaText_theme)
        dpg.bind_item_theme(self.AssignSpeciesCombo, self.magentaText_theme)
        dpg.bind_item_theme(self.AssignCallTypeCombo, self.magentaText_theme)
        dpg.bind_item_theme(self.SpeciesLanguageCombo, orangeText_theme)
        
        with dpg.item_handler_registry(tag="main resize handler") as resize_handler:
            dpg.add_item_resize_handler(callback=self.resize_handler)
        dpg.bind_item_handler_registry(self.mainWindow, "main resize handler")
        
        with dpg.handler_registry(tag="mouse handler") :
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self.zoom_release_handler)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=self.zoom_drag_handler)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Right, callback=self.label_release_handler)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=self.label_drag_handler)
        with dpg.handler_registry(show=True):
            dpg.add_key_press_handler(key=dpg.mvKey_Up, callback=self.UpKey_pressed)
            dpg.add_key_press_handler(key=dpg.mvKey_Down, callback=self.DownKey_pressed)
            dpg.add_key_press_handler(key=dpg.mvKey_Left, callback=self.LeftKey_pressed)
            dpg.add_key_press_handler(key=dpg.mvKey_Right, callback=self.RightKey_pressed)          

        self.classify = Classify()
        if sys.platform.startswith("win"): DragAndDrop.set_drop(self.FileDrop)
        self.SpecDisplay1.dir = config['dir']; self.SpecDisplay1.file = config['file']
        if len(config['echoMeterDir']) > 0:
            self.LoadEchoMeterDir(config['echoMeterDir'])
            rows = self.FilesDF[self.FilesDF['Filename'] == config["file"]]
            if not rows.empty :            
                row = rows.index[0]
                dpg.highlight_table_row(self.FileTable, row, color=[0,100,0])
                self.ScrollToRow(row)
                self.lastRow = row
                self.MultiFile = True; self.resize_handler(0, None, None)
                self.Status(f"'{config['echoMeterDir']}' Echo Meter files, select file") 
        elif self.MultiFile:
            self.LoadBatDetectTable(self.FileTable, self.SpecDisplay1.dir) 
            if len(config["file"]) > 0:
                rows = self.FilesDF[self.FilesDF['Filename'] == config["file"]]
                print(f"MainWindow {config["file"]=}, {rows=}")
                if not rows.empty:
                    row = rows.index[0]
                    try:
                        print(f"MainWindow {config["file"]=}, {row=}")
                        dpg.highlight_table_row(self.FileTable, row, color=[0,100,0]) 
                        self.ScrollToRow(row)
                        self.lastRow = row
                        self.MultiFile = True; self.resize_handler(0, None, None)
                        self.Status(f"'{self.SpecDisplay1.dir}' files, select file") 
                    except: print(colorama.Fore.RED + f"MainWindow {row=} not in self.FileTable?" + colorama.Fore.RESET)
        else:
            self.MultiFile = False; self.resize_handler(0, None, None)
            if config['file'] == EXAMPLE_FILE:
                self.HelpButton_pressed()
            self.Status("Drag Drop file or directory onto App, to load it")
        if len(config["file"]) > 0:
            lastFile = os.path.join(config["dir"], config["file"])
            if os.path.isfile(lastFile):
                self.LoadClassifiedFile(lastFile, self.SpecDisplay1)
            else:
                self.Status(f"LAST FILE '{lastFile}' NO LONGER EXISTS", error=True)
        self.SpeciesLanguageCombo_changed(None, self.SpeciesLanguage, None)
        self.EditModeListbox_changed(None, self.EditMode, None)

    def notify_exception(self, type, value, tb):
        traceback_details = "\n".join(traceback.extract_tb(tb).format())
        msg = f"caller: {' '.join(sys.argv)}\n{type}: {value}\n{traceback_details}"
        print(colorama.Fore.RED + msg + colorama.Fore.RESET)
        self.Status("EXCEPTION see console", error=True)
        
    def FileDialog_Show(self):
        self.FileDialog.Show()

    def FileDialog_Finished(self, f, displayN):
        print(f"FileDialog_Finished {f=} {displayN=}")
        self.LoadFileOrDir(f, displayN)
        self.SetActiveDisplayN(displayN)
        
    def HelpButton_pressed(self):
        webbrowser.open_new_tab(os.path.join("file:", os.getcwd(), "help.html"))
        
    def ScrollToRow(self, row): 
        y = dpg.get_y_scroll(self.FileTable)
        yMax = dpg.get_y_scroll_max(self.FileTable)
        rowSize = yMax/len(self.FilesDF)
        print(f"ScrollToRow {y=} {yMax=} {row=} {len(self.FilesDF)} {rowSize=}")
        if yMax > 0: pxl = row * rowSize
        else: pxl = row * ROW_PXL
        dpg.set_y_scroll(self.FileTable, pxl)

    def resize_handler(self, sender, app_data, user_data):
        config["height"] = dpg.get_viewport_height()
        config["width"] = dpg.get_viewport_width()
        pos = dpg.get_viewport_pos()
        config["x"] = pos[0]
        config["y"] = pos[1]
        #print(f"resize_handler {self.MultiFile=} {config['width']=} {config['height']=} {pos=}")
        
        if self.MultiFile: 
            dpg.configure_item(self.FileTable, show=True)
            topHeight = 0.8 * config["height"]
            dpg.configure_item(self.FileTable, height = config["height"] - topHeight)           
        else:
            dpg.configure_item(self.FileTable, show=False)
            topHeight = config["height"]
        
        if dpg.get_item_configuration(self.SpecDisplay2.topGroup)['show']:
            specHeight = (topHeight - (2*SCROLL_HT + 3*BUTTON_HT + STATUS_HT + HEADER ) * config["scale"] - SPACING *8)//2 
            #print(f"resize_handler 2 displays {specHeight=}")
            dpg.configure_item(self.SpecDisplay1.ampPlot, show=False) 
            dpg.configure_item(self.SpecDisplay2.bottomGroup, show=True)
            dpg.configure_item(self.SpecDisplay1.topGroup, height=specHeight)
            dpg.configure_item(self.SpecDisplay2.topGroup, height=specHeight)
        else:
            specHeight = topHeight - (AMP_HT + SCROLL_HT + 2*BUTTON_HT + STATUS_HT+ HEADER ) * config["scale"] - SPACING *6
            #print(f"resize_handler 1 display {specHeight=}")
            dpg.configure_item(self.SpecDisplay1.ampPlot, show=True)
            dpg.configure_item(self.SpecDisplay1.topGroup, height=specHeight) 
        
    def UpKey_pressed(self, sender, app_data, user_data):
        print(f"UpKey_pressed {self.MultiFile=} {self.lastRow=} {self.ActiveDisplay.dirIndex=}")
        if self.ActiveDisplay == self.SpecDisplay1 and self.MultiFile:
            print(f"UpKey_pressed TableRow")
            if self.lastRow is None or self.lastRow == 0: row = 0
            else: row = self.lastRow -1
            self.ScrollToRow(row)
            user_data = [self.FileTable,  self.FilesDF, row, row]
            print(f"UpKey_pressed {row +1} of {self.NumFiles}")
            self.TableRow_selected(0, [], user_data)
        elif self.ActiveDisplay.dirIndex > 0:
            self.ActiveDisplay.dirIndex -= 1
            self.LoadClassifiedFile(self.ActiveDisplay.dirFiles[self.ActiveDisplay.dirIndex], self.ActiveDisplay)
                
    def DownKey_pressed(self, sender, app_data, user_data):
        print(f"UpKey_pressed {self.MultiFile=} {self.lastRow=} {self.ActiveDisplay.dirIndex=}")
        if self.ActiveDisplay == self.SpecDisplay1 and self.MultiFile:
            print(f"DownKey_pressed TableRow")
            if self.lastRow is None: row = 1
            else: row = self.lastRow +1
            self.ScrollToRow(row)
            if row < len(self.FilesDF):
                user_data = [self.FileTable,  self.FilesDF, row, row]
                print(f"DownKey_pressed {row +1} of {self.NumFiles}")
                self.TableRow_selected(0, [], user_data)      
        elif self.ActiveDisplay.dirIndex < len(self.ActiveDisplay.dirFiles) -1:
            self.ActiveDisplay.dirIndex += 1
            self.LoadClassifiedFile(self.ActiveDisplay.dirFiles[self.ActiveDisplay.dirIndex], self.ActiveDisplay)
            
    def LeftKey_pressed(self, sender, app_data, user_data):
        print(f"LeftKey_pressed {self.Range=}")        
        if self.ActiveDisplay.minT - self.Range < 0: self.ActiveDisplay.maxT = self.Range; self.ActiveDisplay.minT = 0
        else: self.ActiveDisplay.minT -= self.Range; self.ActiveDisplay.maxT -= self.Range
        self.ActiveDisplay.DisplaySpectogram()
        if self.FilesDF is not None: self.FilesDF.loc[self.lastRow, 'minT'] = self.ActiveDisplay.minT
        print(f"LeftKey_pressed {self.ActiveDisplay.minT=}")
        dpg.set_value(self.ActiveDisplay.ScrollBar, self.ActiveDisplay.minT) 

    def RightKey_pressed(self, sender, app_data, user_data):
        print(f"RightKey_pressed {self.Range=}")
        if self.ActiveDisplay.maxT + self.Range > self.ActiveDisplay.duration: 
            self.ActiveDisplay.maxT = self.ActiveDisplay.duration
            if self.ActiveDisplay.minT - self.Range < 0: self.ActiveDisplay.maxT = self.Range; self.ActiveDisplay.minT = 0            
            else: self.ActiveDisplay.minT = self.ActiveDisplay.duration - self.Range
        else: self.ActiveDisplay.minT += self.Range; self.ActiveDisplay.maxT += self.Range
        self.ActiveDisplay.DisplaySpectogram()
        print(f"RightKey_pressed {self.lastRow=} {self.ActiveDisplay.minT}")
        if self.FilesDF is not None: self.FilesDF.loc[self.lastRow, 'minT'] = self.ActiveDisplay.minT
        dpg.set_value(self.ActiveDisplay.ScrollBar, self.ActiveDisplay.minT) 
    
    def zoom_drag_handler(self, sender, app_data, user_data):
        display = self.WhichDisplay()
        if self.ZoomStart is None and display is not None:
            self.ZoomStart = dpg.get_plot_mouse_pos()
            print(f"zoom_drag_handler {self.ZoomStart=}")
    
    def WhichDisplay(self):
        if self.FileDialog.Shown(): return None
        mousePos = dpg.get_mouse_pos()
        plotPos1 = dpg.get_item_pos(self.SpecDisplay1.specPlot)
        plotRect1 = dpg.get_item_rect_size(self.SpecDisplay1.specPlot)
        plotPos2 = dpg.get_item_pos(self.SpecDisplay2.specPlot)
        plotRect2 = dpg.get_item_rect_size(self.SpecDisplay2.specPlot)
        display = None
        if mousePos[1] > plotPos1[1] and mousePos[1] < plotPos1[1] + plotRect1[1] and mousePos[0] > plotPos1[0]: # click on spectrogram1
            display = self.SpecDisplay1
        elif mousePos[1] > plotPos1 [1] and mousePos[1] < plotPos2[1] + plotRect2[1] and mousePos[0] > plotPos2[0]: # click on spectrogram2
            display = self.SpecDisplay2
        return display
        
    
    def zoom_release_handler(self, sender, app_data, user_data):
        display = self.WhichDisplay()
        if self.ZoomStart is not None and display is not None:
            plotPos = dpg.get_plot_mouse_pos()
            if plotPos != self.ZoomStart:
                print(f"zoom_release_handler drag on plot {self.ZoomStart=} {plotPos=}")              
                xLim = dpg.get_axis_limits(display.specXaxis)
                yLim = dpg.get_axis_limits(display.specYaxis)
                a = int(yLim[0] / display.maxF * (display.freqBins-1)); b = int(yLim[1] / display.maxF * (display.freqBins-1))
                print(f"zoom_release_handler on plot {xLim=} {yLim=} {a=} {b=} {display.npPsd.shape=}") 
                peak = (numpy.argmax(display.npPsd[a:b]) + a) / (display.freqBins-1) * display.maxF
                dpg.set_axis_limits(display.psdYaxis, yLim[0], yLim[1])
                dpg.set_axis_limits(display.ampXaxis, xLim[0], xLim[1])               
                dpg.set_item_label(display.psdXaxis, f"Peak {peak:.1f} kHz")
                minPsd =  display.npPsd[a:b].min(); maxPsd = display.npPsd[a:b].max()
                dpg.set_axis_limits(display.psdXaxis, minPsd, maxPsd)
                self.lastMousePlotPos = self.lastMousePos = None
                dpg.configure_item(display.specXaxis, label=f"Time seconds")
                a = int(display.sample_rate * (xLim[0] - display.minT))
                b = int(display.sample_rate * (xLim[1] - display.minT))
                display.ZoomRecording = display.Recording[a:b]
                print(f"zoom_release_handler on plot {xLim=} {yLim=} {display.minT=} {a=} {b=} {display.Recording.shape=} {display.ZoomRecording.shape=}") 
                display.zoomed = True
            self.ZoomStart = None
        elif display is not None:
            mousePos = dpg.get_mouse_pos()
            mousePlotPos = dpg.get_plot_mouse_pos()             
            print(f"click on spectrogram {mousePlotPos=} {display.minT=}")
            if dpg.is_key_down(dpg.mvKey_Prior): 
                self.SpecDisplay1.minF = self.SpecDisplay2.minF = mousePlotPos[1]
                self.SpecDisplay1.DisplaySpectogram()                
                if dpg.is_item_shown(self.SpecDisplay2.topGroup): self.SpecDisplay2.DisplaySpectogram()
            elif dpg.is_key_down(dpg.mvKey_Next): 
                self.SpecDisplay1.maxF = self.SpecDisplay2.maxF = mousePlotPos[1]
                self.SpecDisplay1.DisplaySpectogram()   
                if dpg.is_item_shown(self.SpecDisplay2.topGroup): self.SpecDisplay2.DisplaySpectogram()
            elif self.lastMousePlotPos is not None:
                print(f"second click {self.lastMousePlotPos=} {mousePlotPos=}")
                xLim = dpg.get_axis_limits(display.specXaxis)
                yLim = dpg.get_axis_limits(display.specYaxis)
                print(f"zoom_release_handler not zoom {mousePos=} {xLim=}, {yLim=} == [{display.minT, display.maxT}], [{display.minF,display.maxF}]")
                if display.zoomed and math.isclose(xLim[0],display.minT,abs_tol=0.001) and math.isclose(xLim[1],display.maxT,abs_tol=0.001) and math.isclose(yLim[0],display.minF,abs_tol=1.0) and math.isclose(yLim[1],display.maxF,abs_tol=1.0):
                    # needed to handle unzoom
                    print(f"zoom_release_handler unzoom {mousePos=} {xLim=} {yLim=}")
                    dpg.set_axis_limits(display.psdYaxis, yLim[0], yLim[1])
                    dpg.set_axis_limits(display.ampXaxis, xLim[0], xLim[1])           
                    peak = numpy.argmax(display.npPsd) / (display.freqBins-1) * display.maxF
                    dpg.set_item_label(display.psdXaxis, f"Peak {peak:.1f} kHz")
                    minPsd =  display.npPsd.min(); maxPsd = display.npPsd.max()
                    dpg.set_axis_limits(display.psdXaxis, minPsd, maxPsd)
                    self.lastMousePlotPos = self.lastMousePos = None
                    dpg.configure_item(display.specXaxis, label=f"Time seconds")
                    display.zoomed = False; display.ZoomRecording = None
                else:
                    #display interval between clicks
                    plotInterval = abs(self.lastMousePlotPos[0] - mousePlotPos[0])
                    if plotInterval > 0.002:
                        tMin = min(self.lastMousePlotPos[0], mousePlotPos[0]) # time
                        l = f"{plotInterval*1000:.0f} ms"
                        offset = abs(mousePos[0] - self.lastMousePos[0]) # pixels
                        f = min(mousePlotPos[1], self.lastMousePlotPos[1]) - 15
                        dpg.add_plot_annotation(parent=display.specPlot, label=l, default_value=(tMin, f), offset=(offset, 1), color=[80, 0, 80, 256]) 
                        print(f"zoom_release_handler {plotInterval=} {tMin=} {offset=} on plot")
                    self.lastMousePlotPos = self.lastMousePos = None
            else: 
                self.lastMousePlotPos = mousePlotPos
                self.lastMousePos = mousePos

    def bandpass(self, data: numpy.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
        sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
        print(f"bandpass {len(data)=} second-order filter coefficients {len(sos)=}")
        filtered_data = scipy.signal.sosfiltfilt(sos, data, padtype=None) ### ValueError: The length of the input vector x must be greater than padlen, which is 33
        return filtered_data


    def label_drag_handler(self, sender, app_data, user_data):
        if self.LabelStartPlot is None:
            self.LabelDragDisplay = self.WhichDisplay()
            if self.LabelDragDisplay is not None:
                self.LabelStartPlot = dpg.get_plot_mouse_pos()
                self.LabelStartMouse = dpg.get_mouse_pos()
                if self.EditMode == "None":
                    self.DragStart_MaxT = self.ActiveDisplay.maxT
                    self.DragStart_MinT = self.ActiveDisplay.minT
                    self.dragLastUpdate = time.time()
                print(f"label_drag_handler {self.LabelStartPlot=}")
        else:
            if self.EditMode == "None":
                # Drag scroll display
                xChange = self.LabelStartPlot[0] - dpg.get_plot_mouse_pos()[0]
                if time.time() - self.dragLastUpdate > 0.2: # reduce updates
                    if self.DragStart_MinT + xChange < 0: self.ActiveDisplay.maxT = self.Range; self.ActiveDisplay.minT = 0                
                    if self.DragStart_MaxT + xChange > self.ActiveDisplay.duration: self.ActiveDisplay.maxT = self.ActiveDisplay.duration; self.ActiveDisplay.minT = self.ActiveDisplay.maxT - self.Range
                    else: self.ActiveDisplay.maxT = self.DragStart_MaxT + xChange; self.ActiveDisplay.minT = self.DragStart_MinT + xChange                    
                    self.ActiveDisplay.DisplaySpectogram()
                    print(f"label_drag_handler {self.ActiveDisplay.minT=}")
                    dpg.set_value(self.ActiveDisplay.ScrollBar, self.ActiveDisplay.minT) 
                    self.dragLastUpdate = time.time()
            else:
                # Update rectangle for editing
                self.LabelDragDisplay.RectOnSpec(self.LabelStartPlot, dpg.get_plot_mouse_pos())

    def label_release_handler(self, sender, app_data, user_data):
        if self.EditMode == "None": 
            # finished drag scrolling
            self.LabelStartPlot = None
            return
        display = self.WhichDisplay()
        if self.LabelStartPlot is not None and display is not None:
            plotPos = dpg.get_plot_mouse_pos()
            if plotPos != self.LabelStartPlot:
                if self.AssignSpeciesID is None or self.AssignCallTypeID is None:
                    self.Status("MISSING AN ASSIGN SPECIES VALUE", error=True)
                    return
                self.Status("")
                print(f"label_release_handler on plot {self.LabelStartPlot=} {plotPos=}")
                labelMinT = min(self.LabelStartPlot[0], plotPos[0])
                labelMinF = min(self.LabelStartPlot[1], plotPos[1]) * 1000 # Hz
                labelMaxT = max(self.LabelStartPlot[0], plotPos[0])
                labelMaxF = max(self.LabelStartPlot[1], plotPos[1]) * 1000 # Hz
                
                if self.EditMode == "Source":
                    print("label_release_handler Source")
                    callsCsvPath = os.path.join(self.SpecDisplay1.dir, "ann", f"{self.SpecDisplay1.file}.csv")
                    CallsDF = pandas.read_csv(callsCsvPath)
                    # delete any calls in this rectangle
                    delCallsDF = CallsDF.copy()[(CallsDF['start_time'] > labelMaxT) | (CallsDF['end_time'] < labelMinT) | (CallsDF['low_freq'] > labelMaxF) | (CallsDF['high_freq'] < labelMinF)]
                    callsDeleted = CallsDF.shape[0] - delCallsDF.shape[0]
                    print(f"label_release_handler {callsDeleted=}")                        
                    # insert new call
                    nRow = delCallsDF[delCallsDF['start_time'] > labelMinT].index[0] -1
                    print(f"label_release_handler {nRow=}")                        
                    if 'event' not in delCallsDF.columns :
                        delCallsDF['event'] = 'EchoLocation'
                    new_call = pandas.DataFrame({'det_prob': 0.5,'start_time': float(f"{labelMinT:.4f}"),'end_time':float(f"{labelMaxT:.4f}"),'high_freq':float(f"{labelMaxF:.0f}"),
                        'low_freq':float(f"{labelMinF:.0f}"),'class': self.SpeciesNames["Latin"].iloc[self.AssignSpeciesID], 'class_prob':0.5, 
                        'event': self.CallTypes[self.AssignCallTypeID]}, index=[nRow+1])
                    delCallsDF = pandas.concat([delCallsDF.iloc[:nRow], new_call, delCallsDF.iloc[nRow:]]).reset_index(drop=True)        
                    delCallsDF.to_csv(callsCsvPath, sep=",", index=False)
                    display.ConvertDFtoNP(delCallsDF)
                    display.DisplaySpectogram(UpdateMin= False, sound = False)

                elif self.EditMode == "Train" and display == self.SpecDisplay1:
                    print("label_release_handler Train")
                    id = self.AssignSpeciesID; ct = self.AssignCallTypeID
                    if self.SpecDisplay1.CallsNP is None: # empty array
                        self.SpecDisplay1.CallsNP = numpy.array([[id, labelMinT, labelMaxT, labelMinF/1000, labelMaxF/1000, 1.0, ct]]) 
                    else:
                        self.SpecDisplay1.CallsNP = numpy.append(self.SpecDisplay1.CallsNP, numpy.array([[id, labelMinT, labelMaxT, labelMinF/1000, labelMaxF/1000, 1.0, ct]]), axis = 0)  
                    callsJsonPath = os.path.join(self.SpecDisplay1.dir, "ann", f"{self.SpecDisplay1.file}.json")
                    self.SpecDisplay1.ConvertNPtoJSON(callsJsonPath)
                    display.DisplaySpectogram(UpdateMin= False, sound = False)
                        
                elif self.EditMode == "Species Ref" and display == self.SpecDisplay1:
                    print("label_release_handler Species Ref")
                    # append to source classifier csv at current location using Pandas
                    callsWavPath = os.path.join("SpeciesRef", "ann", f"{self.SpeciesNames[self.FullSpeciesLanguage].iloc[self.AssignSpeciesID]}.wav.csv")
                    callsCsvPath = callsWavPath + ".csv"
                    callLength = labelMaxT - labelMinT; 
                    startSample = int((labelMinT - self.SpecDisplay1.minT) * self.SpecDisplay1.sample_rate)
                    endSample = int((labelMaxT - self.SpecDisplay1.minT) * self.SpecDisplay1.sample_rate)
                    sample = self.SpecDisplay1.Recording[startSample : endSample]
                    callAudio = self.bandpass(sample, [labelMinF, labelMaxF], self.SpecDisplay1.sample_rate) # remove background noise
                    if self.SpecDisplay1.sample_rate != STD_SAMPLING:
                        callAudio = scipy.signal.resample(callAudio, int(len(callAudio) * STD_SAMPLING / self.SpecDisplay1.sample_rate))
                    if os.path.exists(callsCsvPath):
                        callsDF = pandas.read_csv(callsCsvPath)
                        lastCall = callsDF.iloc[-1]
                        nRow = int(lastCall['id']) +1
                        space = 0.05
                        labelMinT = lastCall['end_time'] + space
                        labelMaxT = labelMinT + callLength
                        sampleRate, audio = scipy.io.wavfile.read(callsWavPath)
                        # audio append gap and selected audio
                        silentAudio = numpy.zeros(int(space * STD_SAMPLING))
                        audio = numpy.concatenate((audio, silentAudio, callAudio))
                        scipy.io.wavfile.write(callsWavPath, sampleRate, audio)
                    else:
                        callsDF = pandas.DataFrame(columns = ['id','det_prob','start_time','end_time','high_freq','low_freq','class','class_prob','event'])
                        labelMinT = 0; labelMaxT = callLength
                        nRow = 0
                        scipy.io.wavfile.write(callsWavPath, STD_SAMPLING, callAudio)
                    new_call = [nRow , 0.5,f"{labelMinT:.4f}",f"{labelMaxT:.4f}",f"{labelMaxF:.0f}",f"{labelMinF:.0f}", 
                        self.SpeciesNames["Latin"].iloc[self.AssignSpeciesID], 0.5, self.CallTypes[self.AssignCallTypeID] ]
                    callsDF.loc[len(self.SpecDisplay2.CallsDF)] = new_call
                    callsDF.to_csv(callsCsvPath, sep=",", index=False)

                    self.SpecDisplay2.ConvertDFtoNP(callsDF)
                    self.SpecDisplay2.LoadFile(callsWavPath)
                    self.resize_handler(0, None, None)                    
                self.LabelStartPlot = None   

    def Status(self, txt, error=False):
        if error:
            if self.StatusLabel is not None:
                dpg.bind_item_theme(self.StatusLabel, self.red_align_right)
                dpg.set_item_label(self.StatusLabel, txt)
            chime.error()
        else:
            dpg.bind_item_theme(self.StatusLabel, self.align_right)
            dpg.set_item_label(self.StatusLabel, txt)

    def FileDrop(self, data, keys):
        print(f"FileDrop {data=}, {keys=}")
        displayN = 1
        if len(keys) > 0: 
            print(f"FileDrop self.SpecDisplay2")
            displayN = 2;
            self.SetActiveDisplayN(2)
        elif self.ActiveDisplay == self.SpecDisplay2: displayN = 2;
        f = data[0]
        self.LoadFileOrDir(f, displayN)
    
    def SetActiveDisplay(self, display):
        self.ActiveDisplay = display
        display.ShowActiveDisplay(True)
        if display == self.SpecDisplay1: self.SpecDisplay2.ShowActiveDisplay(False)
        else: self.SpecDisplay1.ShowActiveDisplay(False)

    def SetActiveDisplayN(self, displayN):
        if displayN == 1:
            self.ActiveDisplay = self.SpecDisplay1
            self.SpecDisplay1.ShowActiveDisplay(True)
            self.SpecDisplay2.ShowActiveDisplay(False)
        else:
            self.ActiveDisplay = self.SpecDisplay2
            self.SpecDisplay2.ShowActiveDisplay(True)
            self.SpecDisplay1.ShowActiveDisplay(False)
            
    def activeButton2(self):
        self.SetActiveDisplayN(2)
        
    def activeButton1(self):
        self.SetActiveDisplayN(1)
            
    def LoadFileOrDir(self, f, displayN):
        if displayN == 2:
            print(f"LoadFileOrDir Use Ref SpecDisplay 2")
            if os.path.isdir(f):
                self.Status("Single files only on second display", error=True)
                return
            display = self.SpecDisplay2
            self.MultiFile = False
            dpg.configure_item(self.SpecDisplay2.topGroup, show=True)
            #self.activeButton2()
        else:
            print(f"LoadFileOrDir Use normal Window")
            display= self.SpecDisplay1
            #self.activeButton1()
        dpg.set_value(display.ClassifyLabel, "")
        self.FilesDF = None
        if os.path.isdir(f):
            dirResults_file = os.path.join(f, "BatDetect2 Results.csv")
            display.dir = f
            if display == self.SpecDisplay2:
                self.Status("Single files only on second display", error=True)
                return
            self.MultiFile = True;
            dpg.configure_item(self.SpecDisplay2.topGroup, show=False)
            if os.path.basename(f).startswith("Session_"):
                print(f"FileDrop single echometer directory {f} found")
                self.Status(f"Classifying single echo meter session at {f}")
                self.LoadEchoMeterDir(f)
                config['echoMeterDir'] = f
            elif os.path.isfile(dirResults_file):
                self.LoadBatDetectTable(self.FileTable, f)
                user_data = [self.FileTable,  self.FilesDF, 0, 0]
                self.TableRow_selected(0, [], user_data)
                self.ScrollToRow(0)
                config["echoMeterDir"] = ""
                self.Status(f"{f} files already Classified")
            elif any(x.startswith("Session_") for x in os.listdir(f)):
                print(f"LoadFileOrDir multiple echometer subdirectories {f} found")
                self.Status(f"Classifying multiple Echo Meter sessions at {f}")
                self.LoadEchoMeterDir(f)
                config['echoMeterDir'] = f
            else:
                self.ClassifyDir(f)
                self.Status(f"Classified directory {f}")
            self.resize_handler(0, None, None)
        elif os.path.isfile(f):
            if display == self.SpecDisplay1: self.MultiFile = False; 
            self.LoadClassifiedFile(f, display)
            dpg.configure_item(self.saveMapButton, show=False)  
            config["echoMeterDir"] = ""
            self.resize_handler(0, None, None)
        else: self.Status("NO FILE OR DIRECTOY", error=True)
 
    def LoadEchoMeterDir(self, f):
        self.echoMeter = EchoMeter(self)
        self.LoadGpsTable(self.FileTable, f)                
        dpg.configure_item(self.AssignSpeciesCombo, show=True)
        dpg.configure_item(self.saveMapButton, show=True)   
        self.echoMeterDir = f
        self.Status("All Echo Meter files Classified, select file")                 
    
    def SaveMap_click(self):
        resultFile = self.echoMeter.SaveMap(GpsFilesDF=self.FilesDF)
        self.Status(f"Map saved as {resultFile}")             
            
    def AssignSpeciesCombo_changed(self, sender, app_data, user_data):
        species = app_data;      
        self.AssignSpeciesID = self.SpeciesNames[self.SpeciesNames[self.FullSpeciesLanguage] == species].index.values[0]
        #used in numpy array 
        print(f"AssignSpeciesCombo_changed {self.AssignSpeciesID=} {self.FileTableRow=} {len(config["echoMeterDir"])=}")
        
        if self.FileTableRow is not None and len(config["echoMeterDir"]) > 0:
            row = self.FileTableRow
            self.FilesDF.loc[row, "Species"] = species
            dpg.configure_item(self.GpsSpeciesCells[row], label=species)
            self.echoMeter.SaveEchoMeterDir(self.FilesDF)

    def AssignCallTypeCombo_changed(self, sender, app_data, user_data):
        callType = app_data
        self.AssignCallTypeID = self.CallTypes.index(callType) #used in numpy array
        print(f"AssignCallTypeCombo {callType=} {self.AssignCallTypeID=}")        
                
    def LoadClassifiedFile(self, f, display):
        print(f"LoadClassifiedFile {f=}")
        display.CallsNP = None
        dir = os.path.dirname(f); file = os.path.basename(f)
        callsCsvPath = os.path.join(dir,"ann", file+".csv")
        if not os.path.isfile(callsCsvPath):
            print(f"LoadClassifiedFile {callsCsvPath=} not found")
            self.Status(f"Classifying file {f}")
            results = self.classify.File(f, debug=True)
            if len(results) > 0:
                self.Status(f"Classified file {f}")
        display.LoadClassifiedFile(f, not self.MultiFile)                     
                
    def ClassifyDir(self, dir_path):
        with wakepy.keep.running():
            config["dir"] = dir_path
            classify = Classify()
            files = utils.ListAudioFiles(dir_path)
            # process files
            dpg.delete_item(self.FileTable, children_only=True, slot=0) # remove columns
            dpg.delete_item(self.FileTable, children_only=True, slot=1) # remove rows
            self.SpecDisplay1.FilesDF = self.FilesDF = pandas.DataFrame(columns =["Filename", "Bat Calls"])
            for column in self.FilesDF.columns: 
                dpg.add_table_column(label=column, parent=self.FileTable)

            self.Status(f"Classifying '{os.path.basename(dir_path)}'") 
            for index, audio_file in enumerate(files): 
                result = classify.File(audio_file)
                if len(result) > 0: self.AddToFileTable(audio_file, result)
                self.Status(f"file {index +1} of {len(files)} Classified") 
            dirResults_file = os.path.join(dir_path, "BatDetect2 Results.csv")
            self.FilesDF.to_csv(dirResults_file, index=False)
            self.FilesDF['minT'] = "0.0"
            self.Status(f"'{os.path.basename(dir_path)}' files all Classified, select file") 
            self.lastRow = None
            config["file"] = ""
                
    def EditModeListbox_changed(self, sender, app_data, user_data):
        print(f"EditModeListbox_changed {sender=} {app_data=} {user_data=}")
        if len(config['echoMeterDir']) > 0: showAssign = True
        else:
            self.EditMode = app_data
            if self.EditMode == "None": showAssign = False
            else: 
                showAssign = True
                if self.EditMode == "Train": 
                    self.SpecDisplay1.classifyEnabled = False
                    self.SpecDisplay1.CallsNP = None
                    callsJsonPath = os.path.join(self.SpecDisplay1.dir, "ann", f"{self.SpecDisplay1.file}.json")
                    if os.path.isfile(callsJsonPath):
                        self.SpecDisplay1.ConvertJSONtoNP(callsJsonPath)
                    self.SpecDisplay1.DisplaySpectogram(UpdateMin= False, sound = False)
        dpg.configure_item(self.AssignSpeciesCombo, show=showAssign)
        dpg.configure_item(self.AssignCallTypeCombo, show=showAssign)

    def SpeciesLanguageCombo_changed(self, sender, app_data, user_data):
        print(f"SpeciesLanguageCombo_changed {sender=} {app_data=} {user_data=}")
        self.SpeciesLanguage = self.SpecDisplay1.SpeciesLanguage = self.SpecDisplay2.SpeciesLanguage = app_data
        self.AbbrevSpeciesLanguage = self.FullSpeciesLanguage = self.SpeciesLanguage
        if self.SpeciesLanguage == "LatinAbbrev": self.FullSpeciesLanguage = "Latin"
        elif self.SpeciesLanguage == "EnglishAbbrev": self.FullSpeciesLanguage = "English"
        elif self.SpeciesLanguage == "Latin": self.AbbrevSpeciesLanguage = "LatinAbbrev"
        elif self.SpeciesLanguage == "English": self.AbbrevSpeciesLanguage = "EnglishAbbrev"
        if self.SpeciesLanguage != "None":
            sortedSpecies = list(self.SpeciesNames.sort_values(by=["bat",self.FullSpeciesLanguage], ascending=[False,True])[self.FullSpeciesLanguage])
        else: sortedSpecies = []
        dpg.configure_item(self.AssignSpeciesCombo, items=sortedSpecies)
        self.SpecDisplay1.DisplaySpectogram(UpdateMin= False, sound = False)
        self.SpecDisplay2.DisplaySpectogram(UpdateMin= False, sound = False)
        
    def FilterListBox_changed(self, sender, app_data, user_data):
        self.SpecDisplay1.SpecBlankKfreq = self.SpecDisplay2.SpecBlankKfreq = int(app_data.split()[1])
        self.SpecDisplay1.HighPassFreq = self.SpecDisplay2.HighPassFreq = 1000 * self.SpecDisplay1.SpecBlankKfreq
        self.SpecDisplay2.SpecBlankKfreq = self.SpecDisplay2.SpecBlankKfreq = int(app_data.split()[1])
        self.SpecDisplay2.HighPassFreq = self.SpecDisplay2.HighPassFreq = 1000 * self.SpecDisplay1.SpecBlankKfreq
        print(f"FilterListBox_changed {sender=} {app_data=} {user_data=} {self.SpecDisplay1.SpecBlankKfreq=} {self.SpecDisplay1.HighPassFreq=}")
        self.SpecDisplay1.DisplaySpectogram(UpdateMin= False, sound = False)
        self.SpecDisplay2.DisplaySpectogram(UpdateMin= False, sound = False)
    
    def RangeListbox_changed(self, sender, app_data, user_data):
        self.Range = float(app_data.rstrip("s"))
        self.SpecDisplay1.Range_changed(self.Range)
        self.SpecDisplay2.Range_changed(self.Range)
        
    def TableRow_selected(self, sender, app_data, user_data):
        table = user_data[0]; df = user_data[1]; dfRow = user_data[2]; gRow = user_data[3]
        print(f"TableRow_selected {dfRow=} {gRow=} {self.lastRow=}")
        self.FileTableRow = dfRow
        if len(config["echoMeterDir"]) > 0:
            self.SpecDisplay1.dir = os.path.join(self.echoMeterDir, self.FilesDF.loc[dfRow]["SessionName"])
        try:
            if self.lastRow is not None:
                dpg.unhighlight_table_row(table, self.lastRow)
        except: print(colorama.Fore.RED + "TableRow_selected dpg bug" + colorama.Fore.RESET)
        dpg.highlight_table_row(table, gRow, color=[0,100,0])
        file = df.loc[dfRow]["Filename"] 
        self.lastRow = gRow
        self.LoadClassifiedFile(os.path.join(self.SpecDisplay1.dir,file), self.SpecDisplay1)
                
    def AddToFileTable(self, filename, result):
        file = os.path.basename(filename)
        r = len(self.FilesDF)
        self.FilesDF.loc[r] = [file, result]
        with dpg.table_row(parent=self.FileTable, height=ROW_PXL * config["scale"]):
            dpg.add_selectable(label=file, callback=self.TableRow_selected, span_columns=True, user_data=[self.FileTable, self.FilesDF, r, r])
            amDate = utils.FileDate(result)
            if len(amDate) > 0: result = f"{a} ({amDate})"
            dpg.add_selectable(label=result, callback=self.TableRow_selected, span_columns=True, user_data=[self.FileTable,  self.FilesDF, r, r])
            ysm = dpg.get_y_scroll_max(self.FileTable)
            if ysm > 0: 
                dpg.set_y_scroll(self.FileTable, ysm)
                print(f"AddToFileTable {ysm=}")
                
    def LoadBatDetectTable(self, table, dir):
        config["echoMeterDir"] = ""
        dirResults_file = os.path.join(self.SpecDisplay1.dir, "BatDetect2 Results.csv")
        self.FilesDF = df = pandas.read_csv(dirResults_file)
        nRows, nCols = df.shape
        self.NumFiles = nRows
        self.lastRow = None
        dpg.delete_item(table, children_only=True, slot=0) # remove columns
        dpg.delete_item(table, children_only=True, slot=1) # remove rows
        for column in df.columns: 
            dpg.add_table_column(label=column, parent=table)
        nRow = 0
        for r in range(nRows):
            nCol = 0
            for c in range(nCols):
                col_name = df.columns[c]
                a = df.loc[r][col_name]
                if col_name == "Filename": 
                    if nCol == 0 and not os.path.exists(f"{dir}/{a}"): 
                        nRow -= 1
                        break # ignore file that does not exist
                    tRow = dpg.add_table_row(parent=table, height=ROW_PXL * config["scale"])
                    amDate = utils.FileDate(a)
                    if len(amDate) > 0:
                        a = f"{a} ({amDate})"
                        dpg.add_selectable(label=a, tag=f"row{r}", parent=tRow, callback=self.TableRow_selected, span_columns=True, user_data=[table, df, r, nRow])
                    else:
                        dpg.add_selectable(label=a, tag=f"row{r}", parent=tRow, callback=self.TableRow_selected, span_columns=True, user_data=[table, df, r, nRow])
                else:
                    dpg.add_selectable(label=a, parent=tRow, callback=self.TableRow_selected, span_columns=True, user_data=[table, df, r, nRow])
                nCol += 1
            tRow = None
            nRow += 1
        df['minT'] = 0.0 # extra column to retain position in file

    def LoadGpsTable(self, table, dir_path):
        self.echoMeter = EchoMeter(self)
        self.FilesDF = self.echoMeter.LoadEchoMeterDir(dir_path)
        self.FilesDF['minT'] = "0.0"
        nRows, nCols = self.FilesDF.shape
        self.NumFiles = nRows        
        removeColumns = ['SessionName','Abbrev']
        columns = list(filter(lambda x: x not in removeColumns, self.FilesDF.columns.to_list()))
        
        dpg.delete_item(table, children_only=True, slot=0) # remove columns
        dpg.delete_item(table, children_only=True, slot=1) # remove rows
        for column in columns: 
            if column == "Class": 
                dpg.add_table_column(label="BatDetect2", width_fixed=True, init_width_or_weight=800, parent=table)
            elif column != "minT":
                dpg.add_table_column(label=column, parent=table)
        self.GpsSpeciesCells = []
        nRow = 0
        for r in range(nRows):
            with dpg.table_row(parent=table, height=ROW_PXL * config["scale"]):
                for c in range(nCols):  
                    col_name = self.FilesDF.columns[c]
                    if col_name not in removeColumns and not col_name.startswith('Unnamed'):
                        a = self.FilesDF.loc[r][col_name] 
                        if isinstance(a, float) :
                            f = float(a)
                            if math.isnan(f):
                                a = ''
                        if col_name == "Species":
                            speciesCell = dpg.add_selectable(label=str(a), callback=self.TableRow_selected, span_columns=True, user_data=[table, self.FilesDF, r, nRow])
                            self.GpsSpeciesCells.append(speciesCell)
                            dpg.bind_item_theme(speciesCell, self.magentaText_theme)
                        else: 
                            dpg.add_selectable(label=str(a), callback=self.TableRow_selected, span_columns=True, user_data=[table, self.FilesDF, r, nRow])
            nRow += 1            
    def SaveMap_clicked(self):
        resultFile = self.echoMeter.SaveMap(GpsFilesDF=self.FilesDF)
        self.Status(f"Map saved as {resultFile}") 

if __name__ == '__main__':
    import dearpygui.dearpygui as dpg

    dpg.create_context()
    if sys.platform.startswith("win"): DragAndDrop.initialize()
    config = None
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as jsonfile:
                config = json.load(jsonfile)
                print(f"JSON self.config read successful {config=}")
            dpg.create_viewport(title=TITLE, width=config["width"], height=config["height"],
                x_pos=config["x"], y_pos= config["y"], small_icon='Resources/bat_128px.ico', large_icon='Resources/bat_128px.ico')
            config['font'] = config['font'] /config["scale"]
            config['scale'] = 1
        except Exception as e:
            print(colorama.Fore.RED + f"Error reading {CONFIG_FILE} {e}" + colorama.Fore.RESET)
    if config is None:
        other = None
        for s in get_monitors():
            if s.is_primary == False: other = s        
            print(str(s))
        if other is None:
            font = int(s.width / s.width_mm * 1.3)
            scale = s.width / s.width_mm / 4
            print(f"{font=} {scale=}")
            config = {"echoMeterDir": "", "dir": ".", "file": EXAMPLE_FILE, "minT": 0, "maxT": 1.0, 
                "width":  s.width - 200, "height":  int(s.height - (HEADER + STATUS_HT / 2) * scale), "x": 200 , "y":  0, "font": font, "scale": scale,
                "EditMode": "None", "SpeciesLanguage": "EnglishAbbrev", "Range": "1.0", "SpecBlankKfreq": "10.0", "MultiFile": False}
        else:
            font = int(other.width / other.width_mm* 1.5)
            scale = other.width / other.width_mm / 4
            print(f"{font=} {scale=}")
            config = {"echoMeterDir": "", "dir": ".", "file": EXAMPLE_FILE, "minT": 0, "maxT": 1.0, 
                "width":  other.width, "height":  int(other.height - (HEADER + STATUS_HT / 2) * scale), "x":  other.x, "y":  other.y, "font": font, "scale": scale, 
                "EditMode": "None", "SpeciesLanguage": "EnglishAbbrev", "Range": "1.0", "SpecBlankKfreq": "10.0", "MultiFile": False}
        dpg.create_viewport(title=TITLE, width=config["width"], height=config["height"],
            x_pos=config["x"], y_pos= config["y"], small_icon='Resources/bat_128px.ico', large_icon='Resources/bat_128px.ico')
                   
    main = MainWindow()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(TITLE.replace(" ", ""), True)
    
    while dpg.is_dearpygui_running():
        main.SpecDisplay1.UpdateSoundLine()
        main.SpecDisplay2.UpdateSoundLine()
        dpg.render_dearpygui_frame()

    with open(CONFIG_FILE, "w") as configfile:
        config["minT"] = float(main.SpecDisplay1.minT); config["maxT" ]= float(main.SpecDisplay1.maxT)
        config["dir"] = main.SpecDisplay1.dir; config["file" ]= main.SpecDisplay1.file
        config['width']=int(config["width"]/config["scale"])
        config['height']=int(config["height"]/config["scale"])
        config['x']=int(config["x"]/config["scale"])
        config['y']= int(config["y"]/config["scale"])
        config['font'] = config['font'] /config["scale"]
        config['scale'] = 1
        config["Range"] = main.SpecDisplay1.Range
        config["SpecBlankKfreq"] = main.SpecDisplay1.SpecBlankKfreq
        config["EditMode"] = main.EditMode
        config["SpeciesLanguage"] = main.SpecDisplay1.SpeciesLanguage
        config["MultiFile"] = main.MultiFile
        print(f"exit {config=}") 
        json.dump(config, configfile)
    dpg.destroy_context()