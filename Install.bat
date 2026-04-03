powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
setx /M path "%path%;%USERPROFILE%\.local\bin\"
uv init
uv add dearpygui
uv add numpy
uv add pandas
uv add polars
uv add colorama
uv add soundfile
uv add torch
uv add torchaudio
uv add torchaudio_filters
uv add librosa
uv add folium
uv add screeninfo
uv add DearPyGui_DragAndDrop
uv add chime
uv add wakepy
uv add psutil
uv add sounddevice
uv add mutagen
Resources\create-shortcut --work-dir "%CD%" --icon-file "%CD%\Resources\bat_128px.ico" "%CD%\run.bat" "%USERPROFILE%\Desktop\BatDetect2 DPG.lnk"
uv run bd2Link.py
uv run gui.py
pause
