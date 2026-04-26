import sys, os, re, datetime

def ListAudioFiles(ip_dir: str, TimeExpanded = False):
    matches = []
    for filename in os.listdir(ip_dir):
        filepath = os.path.join(ip_dir, filename)
        if os.path.isdir(filepath): continue
        size = os.path.getsize(filepath)
        f = filename.upper()
        if (f.endswith(".WAV") or f.endswith(".MP3")) and f != 'TEMP.WAV' and size > 0:
            if TimeExpanded == True or not f.endswith("_TE.WAV"):
                matches.append(filepath)
    return matches
   
def FileDate(a):
    if re.match(r'^[0-9]{8}_[0-9]{6}', a):
        #AudioMoth
        return datetime.datetime(int(a[0]+a[1]+a[2]+a[3]), int(a[4]+a[5]), int(a[6]+a[7]), int(a[9]+a[10]), int(a[11]+a[12]), 0)
        return dt.strftime("%d/%m/%Y %H:%M:%S")
        #return f"{a[6]+a[7]}/{a[4]+a[5]}/{a[2]+a[3]} {a[9]+a[10]}:{a[11]+a[12]}"
    _dtSearch = re.search(r'_[0-9]{8}_[0-9]{6}', a)
    if _dtSearch:
        #wildlife acoustics ########_20200501_195401_######.wav ######_ or _###### maybe missing
        a = _dtSearch.group()[1:]
        return datetime.datetime(int(a[0]+a[1]+a[2]+a[3]), int(a[4]+a[5]), int(a[6]+a[7]), int(a[9]+a[10]), int(a[11]+a[12]), int(a[13]+a[14]))
    _dtSearch = re.search(r'_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}', a)
    if _dtSearch:
        #us_??
        a = _dtSearch.group()[1:]            
        print(f"FileDate {a=}")
        return datetime.datetime(int(a[0]+a[1]+a[2]+a[3]), int(a[5]+a[6]), int(a[8]+a[9]), int(a[11]+a[12]), int(a[14]+a[15]), int(a[17]+a[18]))
    else:
        #batlogger samename.xml with datatime
        #Peersonic wav####_YYYY_MM_DD__HH_MM_SS.wav     wav####_ maybe missing
        #Pettersson ###YYYY-MM-DD_HH_MM_SS###.wav       ### maybe missing
        #Apodemus Pippyg and Pipistrelle mini WAV GUANO Metadata 
        #Titley WAV GUANO Metadata 
        return None
            

