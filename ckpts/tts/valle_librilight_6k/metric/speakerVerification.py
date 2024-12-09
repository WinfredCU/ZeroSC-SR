import os
import numpy as np
import torch
from pyannote.audio import Inference
from huggingface_hub import login

# Define the directory containing the audio files
audio_dir = "/home/oem/Winfred/valle/egs/libritts/demo_valle"

# Authenticate with Hugging Face
hf_token = "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"
login(hf_token)

# Function to extract speaker embeddings
def extract_speaker_embeddings(file_path):
    # Load pre-trained speaker embedding model with authentication
    model = Inference("pyannote/embedding", use_auth_token=hf_token, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Extract embeddings
    embeddings = model(file_path)
    
    # Aggregate embeddings (mean pooling)
    embedding = np.mean(embeddings, axis=0)
    
    return embedding

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

# Dictionary to store audio file pairs
audio_pairs = {}

# Populate the dictionary with audio file pairs
for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        prefix = "_".join(file.split("_")[:-1])
        if prefix not in audio_pairs:
            audio_pairs[prefix] = []
        audio_pairs[prefix].append(os.path.join(audio_dir, file))

# Calculate speaker similarity for each pair and store the results
similarity_scores = []
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
        
        print(f"Evaluating speaker similarity for: {prefix}")
        
        # Extract speaker embeddings
        embedding1 = extract_speaker_embeddings(path_original)
        embedding2 = extract_speaker_embeddings(path_clean)
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity(embedding1, embedding2)
        similarity_scores.append(similarity_score)
        print(f"Speaker similarity score for {prefix}: {similarity_score}")

# Calculate the average speaker similarity score
if similarity_scores:
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"Average speaker similarity score: {average_similarity}")
else:
    print("No valid audio pairs found for speaker similarity.")
