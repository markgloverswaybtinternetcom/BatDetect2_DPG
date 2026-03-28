import dearpygui.dearpygui as dpg
import sys, os, time, datetime, psutil, textwrap, soundfile, colorama, WavUtil
from mutagen.mp3 import MP3
if sys.platform.startswith("win"): 
    import win32api

LastRowSelected = None #fixes bug 

class FileDialog(): 
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
        
    def __init__(self, loadCallback):
        self.loadCallback = loadCallback
        dirsTxtFile = os.path.join('Resources', 'BatRecordingDirectories.txt')
        if not os.path.exists(dirsTxtFile):
            f = open(dirsTxtFile, 'w')
            if sys.platform.startswith("win"):             
                f.write(os.path.join(os.path.expanduser("~"), "Downloads"))
                f.write("\n" + os.path.join(os.path.expanduser("~"), "Documents"))
                f.write("\n" + os.path.join(os.path.expanduser("~"), "Desktop"))
            else:
                f.write(os.path.join(os.path.expanduser("~"), "Downloads", "")) # Linux needs separator on end
                f.write("\n" + os.path.join(os.path.expanduser("~"), "Documents", ""))
                f.write("\n" + os.path.join(os.path.expanduser("~"), "Desktop", ""))
            f.close()
        with open(dirsTxtFile, 'r') as file:
            self.RoostDirs = file.read().splitlines()
        drives = self._get_all_drives()
        #print(f"FileDialog {drives=}")
        self.RoostDirs.extend(drives)
        
        self.history = []
        hwidth, hheight, _, hdata = dpg.load_image( "Resources/home.png")
        mfwidth, mfheight, _, mfdata = dpg.load_image("Resources/mini_folder.png")
        with dpg.texture_registry():
            ico_home = dpg.add_static_texture( width=hwidth, height=hheight, default_value=hdata)
            self.img_mini_folder = dpg.add_static_texture( width=mfwidth, height=mfheight, default_value=mfdata)

        with dpg.window(label="File dialog", show=False, modal=True, width=1000, height=-1, no_collapse=True, pos=(0, 100)) as self.window:
            self.CurrentDirectoryText = dpg.add_text("Home")
            with dpg.table(height=525, width=-1, clipper=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, 
                borders_innerV=True, reorderable=True, hideable=True, sortable=True, scrollX=True, scrollY=True) as self.table:
                dpg.add_table_column(label=' ', init_width_or_weight=4)
                dpg.add_table_column(label='Name', init_width_or_weight=110)
                dpg.add_table_column(label='Date', init_width_or_weight=30)
                dpg.add_table_column(label='Length', init_width_or_weight=15)
                dpg.add_table_column(label='Sample Rate', init_width_or_weight=25)
            with dpg.group(horizontal=False):
                with dpg.group(horizontal=True):
                    self.RoostDirButton = dpg.add_image_button(ico_home, label="Bat Recording Dirs", callback=self.DisplayRoost)
                    self.RoostAddDirInputText = dpg.add_input_text(hint="Directory full path")
                    self.RoostAddDirButton = dpg.add_button(label="Add Bat Recording Dir", callback=self.AddRoost_callback)
                    self.UpDirButton = dpg.add_button(arrow=True, direction=dpg.mvDir_Up,label="Up Dir", callback=self.UpDir_callback)
                    self.loadFileButton = dpg.add_button(label="Load WAV / Dir in main", callback=self.LoadFileSelected_callback)
                    self.loadCompareButton = dpg.add_button(label="Load WAV in comparison", callback=self.LoadFileComparison_callback)
                    self.WavMetadataButton = dpg.add_button(label="WAV Metadata in Console", callback=self.WavMetadata_callback)                               
        with dpg.item_handler_registry(tag="file dialog resize handler"):
            dpg.add_item_resize_handler(callback=self.resize_handler)
        dpg.bind_item_handler_registry(self.window, "file dialog resize handler")

        with dpg.theme() as self.redText_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (200, 0, 0, 255), category=dpg.mvThemeCat_Core)                
        with dpg.theme() as self.greenText_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 150, 0, 255), category=dpg.mvThemeCat_Core) 
        with dpg.theme() as greenButton_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 100, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 125, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 150, 0, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10, category=dpg.mvThemeCat_Core)
        with dpg.theme() as gbButton_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 100, 100, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 125, 125, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 150, 150, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10, category=dpg.mvThemeCat_Core)
        with dpg.theme() as navButton_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 75, 100, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 100, 125, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 125, 150, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10, category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.size_alignt:
            with dpg.theme_component(dpg.mvThemeCat_Core):
                dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, x=1, y=.5) 
        dpg.bind_item_theme(self.loadFileButton, greenButton_theme)
        dpg.bind_item_theme(self.loadCompareButton, gbButton_theme)
        dpg.bind_item_theme(self.RoostDirButton, navButton_theme)
        dpg.bind_item_theme(self.RoostAddDirButton, navButton_theme)
        dpg.bind_item_theme(self.UpDirButton, navButton_theme)
 
    def Show(self):
        global LastRowSelected
        print(f"FileDialog Show {len(self.history)=}")
        if len(self.history) == 0: 
            LastRowSelected = self.selectedDir = self.selectedFile = None;
            self.DisplayRoost()
        #else: self.DisplayDir(self.history[-1])
        dpg.configure_item(self.window, show=True) 
    
    def Shown(self):
        return dpg.is_item_shown(self.window)
    
    def _get_all_drives(self):
        all_drives = psutil.disk_partitions()
        if os.name == 'posix':
            drive_list = [drive.mountpoint for drive in all_drives if drive.mountpoint and drive.mountpoint.startswith("/media")]
        else:
            drive_list = [drive.mountpoint for drive in all_drives if drive.mountpoint]
            
        print(f"_get_all_drives {drive_list=}")
        """if os.name == 'posix':
            for device in os.listdir('/dev'):
                if device.startswith("sd") or device.startswith("nvme"):
                    print(f"_get_all_drives {device=}")
                    device_path = f"/dev/{device}"
                    if device_path not in drive_list:
                        drive_list.append(device_path)"""
        return drive_list

    def DisplayRoost(self):
        #print(f"DisplayRoost ")
        dpg.configure_item(self.RoostDirButton, show=False)
        dpg.configure_item(self.RoostAddDirButton, show=True)
        dpg.configure_item(self.RoostAddDirInputText, show=True)
        dpg.configure_item(self.UpDirButton, show=False)
        dpg.configure_item(self.loadFileButton, show=False)
        #dpg.configure_item(self.loadDirButton, show=False)
        dpg.configure_item(self.loadCompareButton, show=False)
        dpg.configure_item(self.WavMetadataButton, show=False)
        self.IsRoostDir = True      
        self.DisplayFiles(self.RoostDirs)
        dpg.set_value(self.CurrentDirectoryText, "Home")
        
    def AddRoost_callback(self, sender, app_data, user_data):
        #print(f"AddRoost_callback {sender=} {app_data=} {user_data=}")
        newDir = dpg.get_value(self.RoostAddDirInputText)
        if os.path.isdir(newDir):
            self.RoostDirs.append(newDir)
            f = open(os.path.join('Resources', 'BatRecordingDirectories.txt'), 'a')
            f.write(f"\n{newDir}")
            f.close()
            self.DisplayRoost()
            dpg.set_value(self.RoostAddDirInputText, "")
            if sys.platform.startswith("win"): 
                cwd = os.getcwd()
                f = open(f"{newDir}/Classify.bat", 'w')
                f.write('title Classify Bat Console')
                f.write(f'\ncall "{cwd}\\.venv\\Scripts\\activate.bat"')
                f.write(f'\npython "{cwd}\\cli.py" %1')
                f.write("\npause")
                f.close()
        
    def DisplayDir(self, dir):
        #print(f"DisplayDir {dir=}")
        if self.IsRoostDir == True:
            dpg.configure_item(self.RoostDirButton, show=True)
            dpg.configure_item(self.RoostAddDirButton, show=False)
            dpg.configure_item(self.RoostAddDirInputText, show=False)
            dpg.configure_item(self.UpDirButton, show=True)
            dpg.configure_item(self.loadFileButton, show=True)
            dpg.configure_item(self.loadCompareButton, show=True) 
            dpg.configure_item(self.WavMetadataButton, show=True)
            self.IsRoostDir = False
        files = []
        for f in os.listdir(dir): 
            if f != "ann": files.append(os.path.join(dir, f))
        self.DisplayFiles(files)
        dpg.set_value(self.CurrentDirectoryText, dir)
        
    def DisplayFiles(self, paths):        
        global LastRowSelected
        dpg.delete_item(self.table, children_only=True, slot=1) # remove rows
        LastRowSelected = None
        nRow = 0
        for entry in paths:
            #print(f"DisplayFiles {entry=}")            
            if entry[-1] == "/": # linux dirs
                entry = entry[:-1]
            name = os.path.basename(entry);
            if os.path.isdir(entry):
                if name.startswith("$") or name.startswith(".") or name == "System Volume Information":
                    continue
                tRow = dpg.add_table_row(parent=self.table)
                dpg.add_image_button(self.img_mini_folder, parent=tRow, height=12, callback=self.TableRow_selected, user_data=[self.table, nRow, entry]) 
                if entry.endswith("\\"): 
                    vInfo = win32api.GetVolumeInformation(entry)
                    name = entry.replace("\\","") 
                    if vInfo[4] == "exFAT" or vInfo[4] == "FAT32": name += " " + vInfo[0] # exchangable drive
                    dpg.add_selectable(label=name, parent=tRow, callback=self.TableRow_selected, user_data=[self.table, nRow, entry])
                else:
                    dpg.add_selectable(label=name, parent=tRow, callback=self.TableRow_selected, user_data=[self.table, nRow, entry])
                    creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(entry))
                    dpg.add_selectable(label=creation_time.strftime("%d/%m/%Y %H:%M"), parent=tRow, callback=self.TableRow_selected, user_data=[self.table, nRow, entry])
                    dpg.add_selectable(label="", parent=tRow, callback=self.TableRow_selected, user_data=[self.table, nRow, entry])
                    if len(self.history) > 0: 
                        selectDir = dpg.add_selectable(label="Select Dir", parent=tRow, callback=self.Dir_selected, user_data=[self.table, nRow, entry])
                        dpg.bind_item_theme(selectDir, self.greenText_theme)
                nRow += 1
            elif os.path.isfile(entry) and (entry.lower().endswith(".wav") or entry.lower().endswith(".mp3")):
                tRow = dpg.add_table_row(parent=self.table)
                dpg.add_selectable(label="", parent=tRow, callback=self.TableRow_selected, user_data=[self.table, nRow, entry])
                file_selectable = dpg.add_selectable(label=name, parent=tRow, callback=self.TableRow_selected, user_data=[self.table, nRow, entry])
                #print(f"DisplayFiles {name=} {nRow=}")
                creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(entry))
                dpg.add_selectable(label=creation_time.strftime("%d/%m/%Y %H:%M"), parent=tRow, callback=self.TableRow_selected, user_data=[self.table, nRow, entry])
                if entry.lower().endswith(".wav"):
                    duration, sample_rate = WavUtil.WavDetails(entry)
                else:
                    audio = MP3(entry)
                    duration = audio.info.length
                    sample_rate = audio.info.bitrate
                if sample_rate > 0:
                    cell_length = dpg.add_selectable(label=f"{duration:.1f} sec", parent=tRow, callback=self.TableRow_selected, user_data=[self.table, nRow, entry])
                    cell_sr = dpg.add_selectable(label=f"{sample_rate / 1000} kHz", parent=tRow, callback=self.TableRow_selected, user_data=[self.table, nRow, entry])
                    dpg.bind_item_theme(cell_length, self.size_alignt)
                    dpg.bind_item_theme(cell_sr, self.size_alignt)                  
                nRow += 1
                
    def TableRow_selected(self, sender, app_data, user_data):
        global LastRowSelected # fixes bug getting old value of self
        table = user_data[0]; nRow = user_data[1]; filepath = user_data[2]
        #print(f"TableRow_selected {nRow=} {filepath=}")
        if LastRowSelected is not None: 
            dpg.unhighlight_table_row(table, LastRowSelected)
        if os.path.isdir(filepath):
            self.history.append(filepath)
            self.DisplayDir(filepath)
        else:
            self.selectedFile = filepath
            dpg.highlight_table_row(table, nRow, color=[0,100,0]) 
            LastRowSelected = nRow 
            # do not add code as will stop selecting rows or buttons
            
    def LoadFileSelected_callback(self):
        print(f"LoadFileSelected_callback {self.selectedFile=}")
        if self.loadCallback is not None:
            dpg.configure_item(self.window, show=False)
            if self.selectedFile is not None:
                self.loadCallback(self.selectedFile, 1)
            elif self.selectedDir is not None:
                self.loadCallback(self.selectedDir, 1)
            self.selectedDir = self.selectedFile = None
    
    def LoadFileComparison_callback(self):
        print(f"LoadFileSelected_callback ")
        if self.loadCallback is not None:
            dpg.configure_item(self.window, show=False)
            if self.selectedFile is not None:
                self.loadCallback(self.selectedFile, 2)
            elif self.selectedDir is not None:
                self.loadCallback(self.selectedDir, 2)
            self.selectedDir = self.selectedFile = None
                
    def Dir_selected(self, sender, app_data, user_data):
        global LastRowSelected # fixes bug getting old value of self
        table = user_data[0]; nRow = user_data[1]; filepath = user_data[2]
        self.selectedDir = filepath
        if LastRowSelected is not None: 
            dpg.unhighlight_table_row(table, LastRowSelected)
        dpg.highlight_table_row(table, nRow, color=[0,100,0])
        LastRowSelected = nRow
        print(f"Dir_selected {self.selectedDir=}")
    
    def UpDir_callback(self):
        #print(f"UpDir_callback ")
        if len(self.history) > 0: self.history.pop()
        if len(self.history) == 0: self.DisplayRoost()
        else: self.DisplayDir(self.history[-1])
    
    def WavMetadata_callback(self):
        if self.selectedFile is not None:
            metadata = WavUtil.parse_metadata(self.selectedFile)
            metadata = metadata.splitlines()
            print(f"\n===== {self.selectedFile} == Metadata =====")
            for line in metadata:
                if len(line) > 60:
                    phrases = line.split(',')
                    for phrase in phrases: print(phrase)
                elif len(line) > 0: print(line)
        
    def resize_handler(self, sender, app_data, user_data):
        windowHeight = dpg.get_item_height(self.window)
        tableHeight = dpg.get_item_height(self.table)
        correctTableHeight = windowHeight - 75
        if abs(tableHeight - correctTableHeight) > 5: 
            #print(f"FileDialog resize_handler {app_data=}")
            dpg.configure_item(self.table, height = correctTableHeight)  