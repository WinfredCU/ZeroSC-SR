import argparse
import logging
import os
from pathlib import Path

import numpy as np
import h5py
import shutil  # Import shutil for file operations


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import get_model


import h5py

import numpy as np

from channel import transmission, compare_and_report
from addnoise import transmission_per_column

# Your specific file path
file_path = '/home/oem/Winfred/valle/examples/libritts/data/tokenized/libritts_encodec_train-clean-100.h5'


# # Open the HDF5 file
# with h5py.File(file_path, 'r') as file:

#     # dataset = file[file_path]
#     # shape = dataset.shape
#     # print('Shape:', shape)

#     # List all groups
#     print("Keys: %s" % file.keys())
#     a_group_key = list(file.keys())[100]

#     # Get the data from the first group
#     data = list(file[a_group_key])

#     # Print the data
#     print(data)

# processed_data = np.concatenate([arr.reshape(1, -1) for arr in data], axis=0)

# # Convert to a PyTorch tensor
# tensor_data = torch.tensor(processed_data, dtype=torch.int32)

# # Reshape tensor
# input_tensor = tensor_data.unsqueeze(0)  # Adds a batch dimension

# # Ensure the tensor is on the correct device
# device = torch.device('cuda:0')
# input_tensor = input_tensor.to(device)

# # Initialize the tokenizer and ensure it is also on the correct device
# audio_tokenizer = AudioTokenizer()
# # Prepare the data for decoding as required by your tokenizer's API
# # Transpose the dimensions if required by your model's input expectations
# # For example, if your model expects (batch, channels, length):
# transposed_input = input_tensor.transpose(2, 1)

# # Decode the data
# samples = audio_tokenizer.decode([(transposed_input, None)])

# print(samples)
# print(samples.shape)

# torchaudio.save("/home/oem/Winfred/valle/egs/libritts/infer/demos/test.wav", samples[0].cpu(), 24000)





##################################################################################################################

def codec2wav(data):
    processed_data = np.concatenate([arr.reshape(1, -1) for arr in data], axis=0)

    # Convert to a PyTorch tensor
    tensor_data = torch.tensor(processed_data, dtype=torch.int32)

    # Reshape tensor
    input_tensor = tensor_data.unsqueeze(0)  # Adds a batch dimension

    # Ensure the tensor is on the correct device
    device = torch.device('cuda:0')
    input_tensor = input_tensor.to(device)

    # Initialize the tokenizer and ensure it is also on the correct device
    audio_tokenizer = AudioTokenizer()
    # Prepare the data for decoding as required by your tokenizer's API
    # Transpose the dimensions if required by your model's input expectations
    # For example, if your model expects (batch, channels, length):
    transposed_input = input_tensor.transpose(2, 1)

    # Decode the data
    samples = audio_tokenizer.decode([(transposed_input, None)])

    # print(samples)
    # print(samples.shape)
    return samples


# #########################################Clean data##########################################################
# N=1
# base_input_path = "/home/oem/Winfred/valle/egs/libritts/download/LibriTTS/train-clean-100"
# base_output_path = "/home/oem/Winfred/valle/egs/libritts/Results"


# with h5py.File(file_path, 'r+') as file:  # Use 'r+' to read/write
#     for key in file.keys():

#         processed_key = key.split('-')[0]

#         print('Processing key:', processed_key)

#         # Build the original and target audio paths based on the processed key
#         parts = processed_key.split('_')
#         speaker_id, chapter_id, utterance_id = parts[0], parts[1], '_'.join(parts[2:])
#         original_audio_path = f"{base_input_path}/{speaker_id}/{chapter_id}/{processed_key}.wav"
#         target_audio_path = f"{base_output_path}/{speaker_id}_{chapter_id}_{utterance_id}_clean.wav"
#         original_target_path = f"{base_output_path}/{speaker_id}_{chapter_id}_{utterance_id}_original.wav"

#         print('Original Audio Path:', original_audio_path)
#         print('Target Audio Path:', target_audio_path)
#         print('Original Target Path:', original_target_path)

#         # Copy the original audio file to the new location
#         shutil.copy(original_audio_path, original_target_path)
#         print(f"Copied original audio to: {original_target_path}")

#         if isinstance(file[key], h5py.Dataset):
#             data = file[key][:]
#             original_shape = data.shape

#             # original_samples = codec2wav(data)

#             processed_data = np.concatenate([arr.reshape(1, -1) for arr in data], axis=0)

#             # Convert to a PyTorch tensor
#             tensor_data = torch.tensor(processed_data, dtype=torch.int32)

#             # Reshape tensor
#             input_tensor = tensor_data.unsqueeze(0)  # Adds a batch dimension

#             # Ensure the tensor is on the correct device
#             device = torch.device('cuda:0')
#             input_tensor = input_tensor.to(device)

#             # Initialize the tokenizer and ensure it is also on the correct device
#             audio_tokenizer = AudioTokenizer()
#             # Prepare the data for decoding as required by your tokenizer's API
#             # Transpose the dimensions if required by your model's input expectations
#             # For example, if your model expects (batch, channels, length):
#             transposed_input = input_tensor.transpose(2, 1)

#             # Decode the data
#             original_samples = audio_tokenizer.decode([(transposed_input, None)])


#             # torchaudio.save("/home/oem/Winfred/valle/egs/libritts/infer/demos/test{}_clean.wav".format(N), original_samples[0].cpu(), 24000)

#             # Save the noisy audio
#             torchaudio.save(target_audio_path, original_samples[0].cpu(), 24000)
#             print(f"File saved: {target_audio_path}")

#             N = N+1




##########################################Noisy Data in total#######################################################
# N=1

# with h5py.File(file_path, 'r+') as file:  # Use 'r+' to read/write
#     for key in file.keys():

#         print('key:',key)
#         if isinstance(file[key], h5py.Dataset):
#             data = file[key][:]
#             original_shape = data.shape

#             print('data:', data)
#             print('data shape:', data.shape)

#             # original_samples = codec2wav(data)

#             ##########################################  Noisy ####################################################
            
#             #### Error occurs in the second loop
#             final_output = transmission(data, SNR_dB=5, gain_mean=5, gain_std=1, channel_type=2)

#             print('final_output:',final_output)


#             reshaped_output = final_output.reshape(original_shape)
#             print('reshaped_output:', reshaped_output)

#             # samples = codec2wav(reshaped_output)


#             processed_data2 = np.concatenate([arr.reshape(1, -1) for arr in reshaped_output], axis=0)

#             # Convert to a PyTorch tensor
#             tensor_data2 = torch.tensor(processed_data2, dtype=torch.int32)

#             # Reshape tensor
#             input_tensor2 = tensor_data2.unsqueeze(0)  # Adds a batch dimension

#             # Ensure the tensor is on the correct device
#             device = torch.device('cuda:0')
#             input_tensor2 = input_tensor2.to(device)

#             # Initialize the tokenizer and ensure it is also on the correct device
#             audio_tokenizer2 = AudioTokenizer()
#             # Prepare the data for decoding as required by your tokenizer's API
#             # Transpose the dimensions if required by your model's input expectations
#             # For example, if your model expects (batch, channels, length):
#             transposed_input2 = input_tensor2.transpose(2, 1)

#             # Decode the data
#             samples = audio_tokenizer2.decode([(transposed_input2, None)])

#             print(samples)
#             print(samples.shape)

#             torchaudio.save("/home/oem/Winfred/valle/egs/libritts/infer/demos/test{}_noisy.wav".format(N), samples[0].cpu(), 24000)

#             print(N)

#             N = N+1

###################################Randomly arrange the order of the matrix and then add noise#############################


N=1
base_input_path = "/home/oem/Winfred/valle/egs/libritts/download/LibriTTS/train-clean-100"
base_output_path = "/home/oem/Winfred/valle/egs/libritts/Results"


with h5py.File(file_path, 'r+') as file:  # Use 'r+' to read/write
    for key in file.keys():

        processed_key = key.split('-')[0]

        print('Processing key:', processed_key)

        # Build the original and target audio paths based on the processed key
        parts = processed_key.split('_')
        speaker_id, chapter_id, utterance_id = parts[0], parts[1], '_'.join(parts[2:])
        original_audio_path = f"{base_input_path}/{speaker_id}/{chapter_id}/{processed_key}.wav"
        target_audio_path = f"{base_output_path}/{speaker_id}_{chapter_id}_{utterance_id}_noisy_random.wav"
        # original_target_path = f"{base_output_path}/{speaker_id}_{chapter_id}_{utterance_id}_original.wav"

        print('Original Audio Path:', original_audio_path)
        print('Target Audio Path:', target_audio_path)
        # print('Original Target Path:', original_target_path)

        # # Copy the original audio file to the new location
        # shutil.copy(original_audio_path, original_target_path)
        # print(f"Copied original audio to: {original_target_path}")



        if isinstance(file[key], h5py.Dataset):
            data = file[key][:]
            original_shape = data.shape

            print('data:', data)
            print('data shape:', data.shape)

            # original_samples = codec2wav(data)

            # Create an index array
            indices = np.arange(data.size)
            np.random.shuffle(indices)

            # Shuffle data based on the randomized indices
            input_shuffled = data.flatten()[indices]

            input_shuffled_originalshape = input_shuffled.reshape(original_shape)


            ##########################################  Noisy ####################################################

            # Define different SNR, gain_mean, and gain_std for each column
            SNR_dBs = [5, 5, 5, 5, 5, 5, 5, 5]
            gain_means = [21, 18, 15, 12, 9, 6, 3, 0]
            gain_stds = [1, 1, 1, 1, 1, 1, 1, 1]

            final_output = transmission_per_column(input_shuffled_originalshape, SNR_dBs, gain_means, gain_stds, channel_type=2)
            
            #### Error occurs in the second loop
            # final_output = transmission(input_shuffled_originalshape, SNR_dB=5, gain_mean=5, gain_std=1, channel_type=2)

            print('###########################After adding noise#############################')

            print('final_output:',final_output)
            print(final_output.shape)


            reshaped_output = final_output.flatten()

            # samples = codec2wav(reshaped_output)


            # Reshape output to original data shape then unshuffle using argsort to revert to original order
            unshuffled_output = reshaped_output[np.argsort(indices)]

            # Reshape to original data shape
            reshaped_unshuffled_output = unshuffled_output.reshape(original_shape)
            print('reshaped_output:', reshaped_unshuffled_output)



            processed_data2 = np.concatenate([arr.reshape(1, -1) for arr in reshaped_unshuffled_output], axis=0)

            # Convert to a PyTorch tensor
            tensor_data2 = torch.tensor(processed_data2, dtype=torch.int32)

            # Reshape tensor
            input_tensor2 = tensor_data2.unsqueeze(0)  # Adds a batch dimension

            # Ensure the tensor is on the correct device
            device = torch.device('cuda:0')
            input_tensor2 = input_tensor2.to(device)

            # Initialize the tokenizer and ensure it is also on the correct device
            audio_tokenizer2 = AudioTokenizer()
            # Prepare the data for decoding as required by your tokenizer's API
            # Transpose the dimensions if required by your model's input expectations
            # For example, if your model expects (batch, channels, length):
            transposed_input2 = input_tensor2.transpose(2, 1)

            # Decode the data
            samples = audio_tokenizer2.decode([(transposed_input2, None)])

            print(samples)
            print(samples.shape)

            # torchaudio.save("/home/oem/Winfred/valle/egs/libritts/infer/demos/test{}_noisy_random.wav".format(N), samples[0].cpu(), 24000)

            # Save the noisy audio
            torchaudio.save(target_audio_path, samples[0].cpu(), 24000)
            print(f"File saved: {target_audio_path}")


            print(N)

            N = N+1

### key: 1034_121119_000002_000001-5242
### original audio path: /home/oem/Winfred/valle/egs/libritts/download/LibriTTS/train-clean-100/1034/121119/1034_121119_000002_000001.wav
### Target audio path: /home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_noisy.wav

# ####################################Sort the order of the matrix based on the column and then add noise#############################


# N=1
# base_input_path = "/home/oem/Winfred/valle/egs/libritts/download/LibriTTS/train-clean-100"
# base_output_path = "/home/oem/Winfred/valle/egs/libritts/Results"

# with h5py.File(file_path, 'r+') as file:  # Use 'r+' to read/write
#     for key in file.keys():

#         processed_key = key.split('-')[0]

#         print('Processing key:', processed_key)

#         # Build the original and target audio paths based on the processed key
#         parts = processed_key.split('_')
#         speaker_id, chapter_id, utterance_id = parts[0], parts[1], '_'.join(parts[2:])
#         original_audio_path = f"{base_input_path}/{speaker_id}/{chapter_id}/{processed_key}.wav"
#         target_audio_path = f"{base_output_path}/{speaker_id}_{chapter_id}_{utterance_id}_nosiy_Sort.wav"

#         print('Original Audio Path:', original_audio_path)
#         print('Target Audio Path:', target_audio_path)


#         if isinstance(file[key], h5py.Dataset):
#             data = file[key][:]
#             original_shape = data.shape

#             print('data:', data)
#             print('data shape:', data.shape)

#             # original_samples = codec2wav(data)



#             ##########################################  Noisy ####################################################

#             # Define different SNR, gain_mean, and gain_std for each column
#             SNR_dBs = [5, 5, 5, 5, 5, 5, 5, 5]
#             gain_means = [21, 18, 15, 12, 9, 6, 3, 0]
#             gain_stds = [1, 1, 1, 1, 1, 1, 1, 1]

#             final_output = transmission_per_column(data, SNR_dBs, gain_means, gain_stds, channel_type=2)
            
#             #### Error occurs in the second loop
#             # final_output = transmission(input_shuffled_originalshape, SNR_dB=5, gain_mean=5, gain_std=1, channel_type=2)

#             print('###########################After adding noise#############################')

#             print('final_output:',final_output)
#             print(final_output.shape)


#             reshaped_output = final_output.reshape(original_shape)

#             # samples = codec2wav(reshaped_output)

#             processed_data2 = np.concatenate([arr.reshape(1, -1) for arr in reshaped_output], axis=0)

#             # Convert to a PyTorch tensor
#             tensor_data2 = torch.tensor(processed_data2, dtype=torch.int32)

#             # Reshape tensor
#             input_tensor2 = tensor_data2.unsqueeze(0)  # Adds a batch dimension

#             # Ensure the tensor is on the correct device
#             device = torch.device('cuda:0')
#             input_tensor2 = input_tensor2.to(device)

#             # Initialize the tokenizer and ensure it is also on the correct device
#             audio_tokenizer2 = AudioTokenizer()
#             # Prepare the data for decoding as required by your tokenizer's API
#             # Transpose the dimensions if required by your model's input expectations
#             # For example, if your model expects (batch, channels, length):
#             transposed_input2 = input_tensor2.transpose(2, 1)

#             # Decode the data
#             samples = audio_tokenizer2.decode([(transposed_input2, None)])

#             print(samples)
#             print(samples.shape)

#             # torchaudio.save("/home/oem/Winfred/valle/egs/libritts/infer/demos/test{}_noisy_sort.wav".format(N), samples[0].cpu(), 24000)


#             # Save the noisy audio
#             torchaudio.save(target_audio_path, samples[0].cpu(), 24000)
#             print(f"File saved: {target_audio_path}")

#             print(N)

#             N = N+1

