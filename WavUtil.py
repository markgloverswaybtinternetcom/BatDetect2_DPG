import os, struct
from typing import Union, BinaryIO, Dict, Optional
from dataclasses import dataclass, field

def WavDetails(filepath):
    with open(filepath, "rb") as wav:
        riff = parse_into_chunks(wav) # Resource Interchange File Format 
        fmt_chunk = riff.subchunks["fmt "]
        data_chunk = riff.subchunks["data"]    
        wav.seek(fmt_chunk.position + 8)
        wav.read(2)  # audio format
        channels = int.from_bytes(wav.read(2), "little")
        sample_rate = int.from_bytes(wav.read(4), "little")
        samples = data_chunk.size // (channels * 2)
        duration = samples / sample_rate
    return duration, sample_rate
        
def parse_metadata(path):
    with open(path, "rb") as wav:
        riff = parse_into_chunks(wav) # Resource Interchange File Format 
        #media_info = get_media_info(wav, riff)
        comment = get_audioMoth_comment(wav, riff)
        guan = get_guan(wav, riff)
        #artist = get_artist(wav, riff)
    return comment + '\n' + guan

CHUNKS_WITH_SUBCHUNKS = ["RIFF", "LIST"]
@dataclass
class Chunk:
    chunk_id: str
    size: int
    position: int
    identifier: Optional[str] = None
    subchunks: Dict[str, "Chunk"] = field(default_factory=dict)

def _get_subchunks(riff: BinaryIO, size: int) -> Dict[str, Chunk]:
    start_position = riff.tell()
    subchunks = {}
    while riff.tell() < start_position + size - 1:
        subchunk = _read_chunk(riff)
        subchunks[subchunk.chunk_id] = subchunk
    return subchunks
    
def parse_into_chunks(riff: BinaryIO):
    riff.seek(0)
    return _read_chunk(riff)

def _read_chunk(riff: BinaryIO):
    position = riff.tell()
    chunk_id = riff.read(4).decode("ascii")
    size = int.from_bytes(riff.read(4), "little")
    identifier = None
    if chunk_id in CHUNKS_WITH_SUBCHUNKS:
        identifier = riff.read(4).decode("ascii")
    chunk = Chunk( chunk_id=chunk_id, size=size, position=position, identifier=identifier)
    if chunk_id in CHUNKS_WITH_SUBCHUNKS:
        chunk.subchunks = _get_subchunks(riff, size - 4)
    else:
        riff.seek(size, os.SEEK_CUR)  
    return chunk

def get_media_info(wav: BinaryIO, chunk: Chunk):
    fmt_chunk = chunk.subchunks["fmt "]
    data_chunk = chunk.subchunks["data"]    
    wav.seek(fmt_chunk.position + 8)
    wav.read(2)  # audio format
    channels = int.from_bytes(wav.read(2), "little")
    samplerate = int.from_bytes(wav.read(4), "little")
    samples = data_chunk.size // (channels * 2)
    duration = samples / samplerate
    return dict(samplerate_hz=samplerate, channels=channels, samples=samples, duration_s=duration)

def get_audioMoth_comment(wav: BinaryIO, chunk: Chunk) -> str:
    if "LIST" not in chunk.subchunks:
        return ""
    list_chunk = chunk.subchunks["LIST"]
    if "ICMT" not in list_chunk.subchunks:
        return ""
    comment_chunk = list_chunk.subchunks["ICMT"]
    wav.seek(comment_chunk.position + 8)
    comment = wav.read(comment_chunk.size - 4).decode("utf-8").strip("\x00")
    return comment

def get_guan(wav: BinaryIO, chunk: Chunk) -> str:
    # guan = Grand Unified Acoustic Notation
    if "guan" not in chunk.subchunks:
        return ""
    guan_chunk = chunk.subchunks["guan"]
    wav.seek(guan_chunk.position + 8)
    guan = wav.read(guan_chunk.size - 4).decode("utf-8").strip("\x00")
    return guan
    
def get_artist(wav: BinaryIO, chunk: Chunk) -> Optional[str]:
    list_chunk = chunk.subchunks.get("LIST")
    if list_chunk is None:
        return ""
    artist_chunk = list_chunk.subchunks.get("IART")
    if artist_chunk is None:
        return ""
    wav.seek(artist_chunk.position + 8)
    artist = wav.read(artist_chunk.size - 4).decode("utf-8").strip("\x00")
    return artist