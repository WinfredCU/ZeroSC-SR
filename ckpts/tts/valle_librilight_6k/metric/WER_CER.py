import whisper
import jiwer
import os

# Load the Whisper model
model = whisper.load_model("medium")

# Define the directory containing the audio files
audio_dir = "/home/oem/Winfred/valle/egs/libritts/demo_valle"

# Function to transcribe audio
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

# Function to calculate WER and CER
def calculate_error_rates(ref_text, hyp_text):
    # Calculate WER
    wer = jiwer.wer(ref_text, hyp_text)
    
    # Calculate CER
    cer = jiwer.cer(ref_text, hyp_text)
    
    return wer, cer

# Dictionary to store audio file pairs
audio_pairs = {}

# Populate the dictionary with audio file pairs
for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        prefix = "_".join(file.split("_")[:-1])
        if prefix not in audio_pairs:
            audio_pairs[prefix] = []
        audio_pairs[prefix].append(os.path.join(audio_dir, file))

# Calculate WER and CER for each pair and store the results
wer_scores = []
cer_scores = []
for prefix, files in audio_pairs.items():
    if len(files) == 2:
        path1, path2 = files
        # Ensure one file has suffix 'ground' and the other has 'valle'
        if path1.endswith("_ground.wav") and path2.endswith("_valle.wav"):
            path_original, path_clean = path1, path2
        elif path1.endswith("_valle.wav") and path2.endswith("_ground.wav"):
            path_original, path_clean = path2, path1
        else:
            continue  # Skip if the suffixes do not match the criteria
        
        print(f"Evaluating WER and CER for: {prefix}")
        
        # Transcribe the audio files
        transcribed_original = transcribe_audio(path_original)
        transcribed_clean = transcribe_audio(path_clean)
        
        # Calculate WER and CER
        wer, cer = calculate_error_rates(transcribed_original, transcribed_clean)
        wer_scores.append(wer)
        cer_scores.append(cer)
        
        print(f"WER for {prefix}: {wer}")
        print(f"CER for {prefix}: {cer}")

# Calculate the average WER and CER scores
if wer_scores:
    average_wer = sum(wer_scores) / len(wer_scores)
    print(f"Average WER: {average_wer}")
else:
    print("No valid audio pairs found for WER.")

if cer_scores:
    average_cer = sum(cer_scores) / len(cer_scores)
    print(f"Average CER: {average_cer}")
else:
    print("No valid audio pairs found for CER.")
