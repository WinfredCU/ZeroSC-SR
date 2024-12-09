import numpy as np
import librosa


path_original = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_original.wav'

path_clean = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_clean.wav'


# path_original = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000001_000001_original.wav'

# path_clean = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000001_000001_clean.wav'


path_noisy_random = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_noisy_random.wav'

path_noisy_sort = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_nosiySort.wav'

def mel_cepstral_distortion(ref_path, deg_path):
    # Load audio files
    y_ref, sr_ref = librosa.load(ref_path, sr=None)
    y_deg, sr_deg = librosa.load(deg_path, sr=None)

    # Check if sampling rates match
    assert sr_ref == sr_deg, "Sampling rates do not match!"

    # Extract MFCCs
    mfcc_ref = librosa.feature.mfcc(y_ref, sr=sr_ref, n_mfcc=13)
    mfcc_deg = librosa.feature.mfcc(y_deg, sr=sr_deg, n_mfcc=13)

    # Calculate the MCD
    mcd = np.sqrt(np.mean(np.sum((mfcc_ref - mfcc_deg)**2, axis=0)))
    return mcd


mcd_score = mel_cepstral_distortion(path_original, path_clean)
print("MCD Score of clean audio:", mcd_score)

mcd_score = mel_cepstral_distortion(path_original, path_noisy_random)
print("MCD Score of noisy_ramdom audio:", mcd_score)

mcd_score = mel_cepstral_distortion(path_original, path_noisy_sort)
print("MCD Score of noisy_sort audio:", mcd_score)
