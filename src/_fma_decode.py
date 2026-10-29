"""Decodificação explícita para o perfil histórico FMA."""
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf

def decode(path, ffmpeg_path):
    if ffmpeg_path is None:
        raise ValueError('O perfil fma-2017 exige ffmpeg_path explícito')
    executable = Path(ffmpeg_path)
    if not executable.is_file():
        raise ValueError('FFmpeg não encontrado')
    info = sf.info(path)
    if info.format != 'MP3':
        raise ValueError('Perfil fma-2017 validado somente para MP3')
    command = [str(executable.resolve()), '-v', 'error', '-c:a', 'mp3', '-i', str(path),
               '-map', '0:a:0', '-f', 's16le', '-c:a', 'pcm_s16le', 'pipe:1']
    output = subprocess.run(command, check=True, capture_output=True)
    channels = np.frombuffer(output.stdout, dtype='<i2').reshape(-1, info.channels)
    # Em MP3, info.frames pode ser uma estimativa do cabeçalho. A quantidade
    # efetivamente decodificada é a referência, como nos diagnósticos anteriores.
    if not len(channels):
        raise ValueError('MP3 não contém amostras decodificadas')
    return np.mean(channels.astype(np.float32).T / 32768, axis=0), info.samplerate
