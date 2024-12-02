import h5py
import numpy as np

from models.tts.valle.channel.channel import transmission, compare_and_report

# file_path = '/home/oem/Winfred/valle/examples/libritts/data/tokenized/libritts_encodec_train-clean-100.h5'

# ################################## Modify All h5py together #############################################################

# N = 1
# with h5py.File(file_path, 'r+') as file:  # Use 'r+' to read/write
#     for key in file.keys():

#         print('key:', key)
#         if isinstance(file[key], h5py.Dataset):
#             data = file[key][:]
#             original_shape = data.shape

#             print('data:', data)
#             print('data shape:', data.shape)

#             # final_output = transmission(data, SNR_dB=5, gain_mean=10, gain_std=1, channel_type=2)
#             # input_data = np.array(data).flatten()

#             # print('input:', input_data)
#             # print("Final output:", final_output)
#             # print("Comparison report:", compare_and_report(input_data, final_output))

#             # reshaped_output = final_output.reshape(original_shape)
#             # print('reshaped_output:', reshaped_output)

#             print(N)
#             N = N + 1



################################### Modify Each column with different SNR #############################################################


def transmission_per_column(data, SNR_dBs, gain_means, gain_stds, channel_type=2):
    num_rows, num_cols = data.shape
    # Ensure we have lists of SNR_dB, gain_mean, gain_std for each column
    assert len(SNR_dBs) == len(gain_means) == len(gain_stds) == num_cols, "Parameter lists must match the number of columns"
    
    combined_output = np.zeros_like(data, dtype=np.float32)  # Initialize combined output array
    
    # Process each column individually with its corresponding parameters
    for col in range(num_cols):
        column_data = data[:, col].reshape(-1, 1)  # Reshape to fit transmission function's input format
        # Use individual SNR, gain_mean, and gain_std for this column
        output = transmission(column_data, SNR_dBs[col], gain_means[col], gain_stds[col], channel_type)
        reshaped_output = output.reshape(-1, 1)  # Reshape back to original column shape
        combined_output[:, col] = reshaped_output.flatten()  # Store the processed column in combined output
    
    return combined_output

# # Example usage with an h5py file
# file_path = '/home/oem/Winfred/valle/examples/libritts/data/tokenized/libritts_encodec_train-clean-100.h5'

# N = 1
# output_file_path = '/home/oem/Winfred/valle/examples/libritts/data/tokenized/libritts_encodec_train-clean-100_NoisyColumn1.h5'
# with h5py.File(file_path, 'r+') as file:  # Use 'r+' to read/write
#     for key in file.keys():
#         if isinstance(file[key], h5py.Dataset):
#             data = file[key][:]
#             original_shape = data.shape

#             print(original_shape)
#             print(data)

#             # Define different SNR, gain_mean, and gain_std for each column
#             SNR_dBs = [5, 5, 5, 5, 5, 5, 5, 5]
#             gain_means = [21, 18, 15, 12, 9, 6, 3, 0]
#             gain_stds = [1, 1, 1, 1, 1, 1, 1, 1]

#             final_output = transmission_per_column(data, SNR_dBs, gain_means, gain_stds, channel_type=2)
#             input_data = np.array(data).flatten()

#             reshaped_output = final_output.reshape(original_shape)  # Reshape to match the original shape

#             print('Input data:\n', data)
#             print("Final output:\n", reshaped_output)
#             print("Comparison report:", compare_and_report(data, reshaped_output))

#             print(N)
            
#             N = N + 1

#             # Optionally, write the reshaped_output back to the HDF5 file
#             # Ensure the dataset you're writing to is properly sized to accommodate the reshaped_output
#             # file[key][:] = reshaped_output

#             # # store in another h5 file
#             # with h5py.File(output_file_path, 'a') as outfile:  # Open the output file in append mode
#             #     # Check if the dataset already exists, and if so, delete it
#             #     if key in outfile:
#             #         del outfile[key]
#             #     # Create a new dataset in the output file and write the processed data
#             #     outfile.create_dataset(key, data=final_output)





# # Example usage
# input_indices_matrix = np.random.randint(0, 1024, size=(100, 8))  # Sample input
# final_output = transmission(input_indices_matrix, SNR_dB=5, gain_mean=10, gain_std=1, channel_type=2)

# input_data = np.array(input_indices_matrix).flatten()
# print('input:', input_data)
# print("Final output:", final_output)
# print("Comparison report:", compare_and_report(input_data, final_output))













