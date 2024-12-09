from pystoi import stoi
import soundfile as sf
import numpy as np

# path_original = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_original.wav'

# path_clean = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_clean.wav'


# path_original = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000001_000001_original.wav'

# path_clean = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000001_000001_clean.wav'


# path_noisy_random = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_noisy_random.wav'

# path_noisy_sort = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_nosiySort.wav'

# path_predict_clean = '/home/oem/Winfred/valle/egs/libritts/infer/demos/infer_clean_3s.wav'



# path_original = '/home/oem/Winfred/valle/egs/libritts/infer/demos/1034_121119_000002_000001_10s.wav'

# path_clean = '/home/oem/Winfred/valle/egs/libritts/infer/demos/infer_clean_3s.wav' 

# path_noisy_random = '/home/oem/Winfred/valle/egs/libritts/infer/demos/infer_noise_random_3s.wav'

# path_noisy_sort = '/home/oem/Winfred/valle/egs/libritts/infer/demos/infer_noise_sort_3s.wav'


path_original = '/home/oem/Winfred/valle/egs/libritts/infer/demos/8455_210777_000067_000000_clean.wav'

path_clean = '/home/oem/Winfred/valle/egs/libritts/prompts/8455_210777_000003_000000.wav' 

# path_noisy_random = '/home/oem/Winfred/valle/egs/libritts/infer/demos/infer_noise_random_3s.wav'

# path_noisy_sort = '/home/oem/Winfred/valle/egs/libritts/infer/demos/infer_noise_sort_3s.wav'

# # Load your audio files
# ref, sr_ref = sf.read(path_original)
# deg, sr_deg = sf.read(path_clean)

# # Ensure both files have the same sample rate
# assert sr_ref == sr_deg, "Sample rates do not match!"

# # Calculate the STOI
# intelligibility_index = stoi(ref, deg, sr_ref, extended=False)
# print("STOI intelligibility index:", intelligibility_index)



def load_and_prepare_audio(file_path, target_length):
    audio, sr = sf.read(file_path)
    current_length = len(audio)
    if current_length < target_length:
        # Pad the audio if it is shorter than the target length
        padding = np.zeros(target_length - current_length)
        audio = np.concatenate((audio, padding))
    elif current_length > target_length:
        # Optionally truncate the audio if it is longer than the target length
        audio = audio[:target_length]
    return audio, sr


def calculate_stoi(path_original, path_clean):
    # Load the original and degraded audio files
    ref, sr_ref = sf.read(path_original)
    deg, sr_deg = sf.read(path_clean)

    # Ensure both files have the same sample rate
    assert sr_ref == sr_deg, "Sample rates do not match!"

    # Prepare audio files
    max_length = max(len(ref), len(deg))
    ref, _ = load_and_prepare_audio(path_original, max_length)
    deg, _ = load_and_prepare_audio(path_clean, max_length)

    # Calculate the STOI
    intelligibility_index = stoi(ref, deg, sr_ref, extended=False)
    return intelligibility_index

# Example usage
stoi_index = calculate_stoi(path_original, path_clean)
print("STOI intelligibility index of clean audio:", stoi_index)

# stoi_index = calculate_stoi(path_original, path_noisy_random)
# print("STOI intelligibility index of noisy_random audio:", stoi_index)

# stoi_index = calculate_stoi(path_original, path_noisy_sort)
# print("STOI intelligibility index of noisy_sort audio:", stoi_index)




# ref, sr_ref = sf.read(path_original)
# deg, sr_deg = sf.read(path_clean)

# # Ensure both files have the same sample rate
# assert sr_ref == sr_deg, "Sample rates do not match!"

# # Prepare audio files
# max_length = max(len(ref), len(deg))
# ref, _ = load_and_prepare_audio(path_original, max_length)
# deg, _ = load_and_prepare_audio(path_clean, max_length)

# # Calculate the STOI
# intelligibility_index = stoi(ref, deg, sr_ref, extended=False)
# print("STOI intelligibility index:", intelligibility_index)
