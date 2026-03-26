#!/bin/bash
sudo snap install astral-uv --classic
uv init
uv add dearpygui
uv add numpy
uv add pandas
uv add colorama
uv add soundfile
uv add sounddevice
uv add torch
uv add torchaudio
uv add torchaudio_filters
uv add librosa
uv add folium
uv add screeninfo
uv add chime
uv add wakepy
uv add psutil
uv add mutagen
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
chmod u+x ./run.sh
echo '[Desktop Entry]' > ~/Desktop/dpg.desktop
echo 'Name=BatDetect2_DPG' >> ~/Desktop/dpg.desktop
echo 'Type=Application' >> ~/Desktop/dpg.desktop
echo "Exec=$(pwd)/run.sh" >> ~/Desktop/dpg.desktop
echo "Icon=$(pwd)/Resources/bat_128px.ico" >> ~/Desktop/dpg.desktop
echo 'Terminal=true' >> ~/Desktop/dpg.desktop
chmod u+x ~/Desktop/dpg.desktop
echo "Need to Allow Launching on Desktop shortcut file dpg.desktop"
uv run gui.py
/bin/bash
