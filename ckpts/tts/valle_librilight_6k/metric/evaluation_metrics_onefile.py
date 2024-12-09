import os
import whisper
import jiwer
import soundfile as sf
import numpy as np
from scipy.signal import resample
from pystoi.stoi import stoi
from pesq import pesq
import librosa
import torch
from pyannote.audio import Inference
from huggingface_hub import login
import re

# Define the directory containing the audio files
# audio_dir = "/home/oem/Winfred/valle/egs/libritts/demo_valle"

# Load the Whisper model
whisper_model = whisper.load_model("medium")

# Authenticate with Hugging Face
hf_token = "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"
login(hf_token)

# Function to transcribe audio
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove hyphens
    text = text.replace('-', ' ')
    # Remove punctuation except hyphens
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to calculate WER and CER
def calculate_error_rates(ref_text, hyp_text):

    print(ref_text)
    print(hyp_text)

    ref_text_norm = normalize_text(ref_text)
    hyp_text_norm = normalize_text(hyp_text)

    print(ref_text_norm)
    print(hyp_text_norm)

    wer = jiwer.wer(ref_text_norm, hyp_text_norm)
    cer = jiwer.cer(ref_text_norm, hyp_text_norm)
    return wer, cer

# Function to load and prepare audio
def load_and_prepare_audio(file_path, target_length):
    audio, sr = sf.read(file_path)
    current_length = len(audio)
    if current_length < target_length:
        padding = np.zeros(target_length - current_length)
        audio = np.concatenate((audio, padding))
    elif current_length > target_length:
        audio = audio[:target_length]
    return audio, sr

# Function to downsample audio to a target sample rate
def downsample_audio(audio, original_sr, target_sr):
    num_samples = int(len(audio) * float(target_sr) / original_sr)
    downsampled_audio = resample(audio, num_samples)
    return downsampled_audio, target_sr

# Function to calculate STOI for a pair of files
def calculate_stoi(path_original, path_clean):
    ref, sr_ref = sf.read(path_original)
    deg, sr_deg = sf.read(path_clean)
    if sr_ref != sr_deg:
        if sr_ref > sr_deg:
            ref, sr_ref = downsample_audio(ref, sr_ref, sr_deg)
        else:
            deg, sr_deg = downsample_audio(deg, sr_deg, sr_ref)
    max_length = max(len(ref), len(deg))
    ref, _ = load_and_prepare_audio(path_original, max_length)
    deg, _ = load_and_prepare_audio(path_clean, max_length)
    intelligibility_index = stoi(ref, deg, sr_ref, extended=False)
    return intelligibility_index

# Function to calculate PESQ for a pair of files
def calculate_pesq(path_original, path_clean):
    ref, sr_ref = sf.read(path_original)
    deg, sr_deg = sf.read(path_clean)
    if sr_ref >= 16000 and sr_deg >= 16000:
        target_sr = 16000
        mode = 'wb'
    elif sr_ref > 8000 and sr_ref < 16000 and sr_deg > 8000 and sr_deg < 16000:
        target_sr = 8000
        mode = 'nb'
    else:
        target_sr = min(sr_ref, sr_deg)
        mode = 'nb' if target_sr <= 8000 else 'wb'
    if sr_ref != target_sr:
        ref, _ = downsample_audio(ref, sr_ref, target_sr)
    if sr_deg != target_sr:
        deg, _ = downsample_audio(deg, sr_deg, target_sr)
    pesq_score = pesq(target_sr, ref, deg, mode)
    return pesq_score

# Function to calculate MCD for a pair of files
def mel_cepstral_distortion(ref_path, deg_path):
    y_ref, sr_ref = librosa.load(ref_path, sr=None)
    y_deg = librosa.load(deg_path, sr=None)[0]
    target_sr = min(sr_ref, sr_ref)
    if sr_ref != target_sr:
        y_ref = librosa.resample(y_ref, sr_ref, target_sr)
    if sr_ref != target_sr:
        y_deg = librosa.resample(y_deg, sr_ref, target_sr)
    mfcc_ref = librosa.feature.mfcc(y_ref, sr=target_sr, n_mfcc=13)
    mfcc_deg = librosa.feature.mfcc(y_deg, sr=target_sr, n_mfcc=13)
    min_length = min(mfcc_ref.shape[1], mfcc_deg.shape[1])
    mfcc_ref = mfcc_ref[:, :min_length]
    mfcc_deg = mfcc_deg[:, :min_length]
    mcd = np.sqrt(np.mean(np.sum((mfcc_ref - mfcc_deg) ** 2, axis=0)))
    return mcd

# Function to extract speaker embeddings
def extract_speaker_embeddings(file_path):
    model = Inference("pyannote/embedding", use_auth_token=hf_token, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    embeddings = model(file_path)
    embedding = np.mean(embeddings, axis=0)
    return embedding

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity


#############################################################################


# path_original = "/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/1089_134686_000001_000001_9s.wav"
# path_clean = "/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/clean.wav"

path_original = '/home/oem/Winfred/Amphion/bins/demo_VALLE/6829_68771_000010_000009.wav'

# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/BadStage1.wav'
# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/BadStage2.wav'
# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/BadStage3.wav'
# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/BadStage4.wav'
# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/BadStage5.wav'
# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/BadStage6.wav'
# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/BadStage7.wav'
# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/BadStage8.wav'

# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/noise_random.wav'
# path_clean = '/home/oem/Winfred/Amphion/ckpts/tts/valle_librilight_6k/result/single/noise_sort.wav'

# path_clean = '/home/oem/Winfred/Amphion/bins/demo_VALLE/6829_68771_000010_000009_SortMatch.wav'
path_clean = '/home/oem/Winfred/Amphion/bins/demo_VALLE/6829_68771_000010_000009_RandomMatch.wav'


# Transcribe the audio files
transcribed_original = transcribe_audio(path_original)
transcribed_clean = transcribe_audio(path_clean)

# print(transcribed_original)
# print(transcribed_clean)

# Calculate WER and CER
wer, cer = calculate_error_rates(transcribed_original, transcribed_clean)

print(f"WER : {wer}")
print(f"CER : {cer}")

# Calculate STOI
stoi_score = calculate_stoi(path_original, path_clean)
print(f"STOI intelligibility index for: {stoi_score}")

# Calculate PESQ
pesq_score = calculate_pesq(path_original, path_clean)
print(f"PESQ score: {pesq_score}")

# Calculate MCD
mcd_score = mel_cepstral_distortion(path_original, path_clean)
print(f"MCD score: {mcd_score}")

# Extract speaker embeddings and calculate cosine similarity
embedding1 = extract_speaker_embeddings(path_original)
embedding2 = extract_speaker_embeddings(path_clean)
similarity_score = cosine_similarity(embedding1, embedding2)
print(f"Speaker similarity score for: {similarity_score}")

