
path_original = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_original.wav'

path_clean = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_clean.wav'


# path_original = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000001_000001_original.wav'

# path_clean = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000001_000001_clean.wav'


path_noisy_random = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_noisy_random.wav'

path_noisy_sort = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_nosiySort.wav'


import librosa
import soundfile as sf
from pesq import pesq

def resample_audio(input_path, target_sr):
    # Load audio file
    audio, sr = librosa.load(input_path, sr=None)  # Load at original sr
    # Resample audio
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio_resampled, target_sr

def calculate_pesq(ref_path, deg_path, target_sr, mode):
    # Resample reference and degraded audio files
    ref, _ = resample_audio(ref_path, target_sr)
    deg, _ = resample_audio(deg_path, target_sr)
    
    # Calculate PESQ score
    score = pesq(target_sr, ref, deg, mode)
    return score

# Example usage

target_sr = 16000  # Set this to 8000 for NB or 16000 for WB
mode = 'wb'  # 'nb' for narrowband, 'wb' for wideband

score_clean = calculate_pesq(path_original, path_clean, target_sr, mode)
print("PESQ score of clean audio:", score_clean)

score_nosiy_random = calculate_pesq(path_original, path_noisy_random, target_sr, mode)
print("PESQ score of noisy_random audio:", score_nosiy_random)

score_noisy_sort = calculate_pesq(path_original, path_noisy_sort, target_sr, mode)
print("PESQ score of noisy_sort audio:", score_noisy_sort)

