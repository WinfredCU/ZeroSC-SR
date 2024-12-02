import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message


def index_to_binary(encoded_indices, bits_per_index):
    # Convert indices to binary strings and then to a list of bits
    binary_strings = [format(index, '0{}b'.format(bits_per_index)) for index in encoded_indices.flatten()]
    binary_array = np.array([[int(bit) for bit in string] for string in binary_strings]).flatten()
    return binary_array


def index_to_binary(indices, bits_per_index):
   binary_strings = [format(index, '0{}b'.format(bits_per_index)) for index in indices.flatten()]
   binary_array = np.array([[int(bit) for bit in string] for string in binary_strings]).flatten()
   return binary_array


def pad_binary_data_for_ldpc(binary_data, bits_per_stream=10):
    """
    Pads binary data where each 10-bit stream is explicitly padded to 11 bits 
    by adding a 0 at the end, then all streams are combined into a single array.
    
    :param binary_data: The binary data array consisting of multiple 10-bit streams.
    :param bits_per_stream: Number of bits in each original stream before padding. Default is 10.
    :return: A single NumPy array with each 10-bit stream padded to 11 bits.
    """
    # Reshape the binary data into chunks of 10 bits each
    num_streams = binary_data.shape[0] // bits_per_stream
    reshaped_data = binary_data[:num_streams * bits_per_stream].reshape(-1, bits_per_stream)
    
    # Create padding of one zero bit for each stream
    padding = np.zeros((num_streams, 1), dtype=reshaped_data.dtype)
    
    # Combine the original bits with the padding
    padded_data = np.hstack((reshaped_data, padding))
    
    # Flatten the padded data back into a single array
    padded_binary_data = padded_data.flatten()
    
    return padded_binary_data



def cut_binary_data_from_ldpc(padded_binary_data, bits_per_stream=11):
    """
    Cuts padded binary data where each 11-bit stream is truncated to 10 bits 
    by removing the last bit, then all streams are combined into a single array.
    
    :param padded_binary_data: The padded binary data array consisting of multiple 11-bit streams.
    :param bits_per_stream: Number of bits in each padded stream before cutting. Default is 11.
    :return: A single NumPy array with each 11-bit stream cut to 10 bits.
    """
    # Calculate the number of 11-bit streams
    num_streams = padded_binary_data.shape[0] // bits_per_stream
    
    # Reshape the padded binary data into chunks of 11 bits each
    reshaped_data = padded_binary_data[:num_streams * bits_per_stream].reshape(-1, bits_per_stream)
    
    # Remove the last bit from each stream
    cut_data = reshaped_data[:, :-1]
    
    # Flatten the cut data back into a single array
    cut_binary_data = cut_data.flatten()
    
    return cut_binary_data


def ldpc_encode(binary_data, H, G):
    # Assumes binary_data is already padded to match the LDPC block size
    # print(G.shape)
    # print(binary_data.shape)
    binary_data_reshaped = binary_data.reshape(-1, G.shape[1])  # Adjust based on the generator matrix's dimensions
    encoded_data = np.array([encode(G, block, snr=1000) for block in binary_data_reshaped])  # Encode each block
    return encoded_data.flatten()

    

def ldpc_decode(received_data, H, G, snr):
    """
    Decodes multiple blocks of LDPC-encoded data.
    
    :param received_data: The received data array, consisting of multiple encoded blocks.
    :param H: The parity check matrix.
    :param G: The generator matrix.
    :param snr: The signal-to-noise ratio (SNR) of the received data.
    :return: The concatenated decoded binary data of all blocks.
    """
    block_size = G.shape[0]  # Assuming the size of each block matches the generator matrix's column count

    # print('block_size:', block_size)
    num_blocks = received_data.shape[0] // block_size
    
    # Initialize an empty list to hold decoded blocks
    decoded_blocks = []
    
    # Process each block
    for i in range(num_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        block = received_data[start:end]

        # Decode the received data
        decoded_block = decode(H, block, snr)

        decoded_data = decoded_block[0:G.shape[1]]

        decoded_blocks.append(decoded_data)

    # print(decoded_blocks)

    decoded_message = np.concatenate(decoded_blocks)
    
    return decoded_message


def compare_and_report(array1, array2):
    if array1.shape != array2.shape:
        return "Arrays have different shapes."
    differences = np.where(array1 != array2)
    if differences[0].shape[0] > 0:
        return f"Differences at positions {differences}"
    else:
        return "Arrays are identical."


def generate_256qam_constellation():
    """
    Generate a basic 256-QAM constellation with normalized energy.
    This function is illustrative and uses a simple square grid constellation.
    """
    # Generate 16 points for both I and Q to form a 256 point grid
    points = np.linspace(-15, 15, 16)
    constellation = np.array([complex(i, q) for i in points for q in points])
    
    # Normalize the constellation average power to 1
    average_power = np.mean(np.abs(constellation)**2)
    constellation /= np.sqrt(average_power)
    
    return constellation

def pad_data_for_256qam(ldpc_encoded_data):
    """
    Pad the LDPC-encoded data to ensure its length is a multiple of 8.
    """
    padding_length = (-ldpc_encoded_data.shape[0]) % 8
    if padding_length > 0:
        ldpc_encoded_data = np.pad(ldpc_encoded_data, (0, padding_length), 'constant', constant_values=(0,))
    return ldpc_encoded_data

def qam256_modulation(ldpc_encoded_data, constellation):
    """
    Modulate LDPC-encoded data using 256-QAM.
    
    :param ldpc_encoded_data: LDPC-encoded data array.
    :param constellation: The 256-QAM constellation.
    :return: Array of 256-QAM modulated complex symbols.
    """
    # Ensure data is padded to a multiple of 8
    ldpc_encoded_data = pad_data_for_256qam(ldpc_encoded_data)
    
    # Convert bipolar (-1, 1) to binary (0, 1)
    binary_data = (ldpc_encoded_data + 1) // 2
    
    # Reshape binary data into 8-bit chunks
    binary_data = binary_data.reshape(-1, 8)
    
    # Convert binary chunks to decimal indices
    indices = np.dot(binary_data, 1 << np.arange(8)[::-1])
    
    # Map indices to constellation points
    qam_symbols = constellation[indices]
    
    return qam_symbols


def convert_bipolar_to_binary(ldpc_encoded_data):
    # Convert bipolar (-1, 1) to binary (0, 1)
    binary_data = (ldpc_encoded_data > 0).astype(int)
    return binary_data


def convert_bipolar_to_binary(ldpc_encoded_data):
    # Convert bipolar (-1, 1) to binary (0, 1)
    binary_data = (ldpc_encoded_data > 0).astype(int)
    return binary_data

def qam256_demodulation_and_remove_padding(qam_symbols, constellation, original_data_length):
    """
    Demodulate 256-QAM symbols back to bipolar LDPC-encoded data and remove any padding zeros.
    
    :param qam_symbols: Array of 256-QAM modulated complex symbols.
    :param constellation: The 256-QAM constellation.
    :param original_data_length: The length of the original LDPC-encoded data before padding.
    :return: Demodulated LDPC-encoded data in bipolar format with padding removed.
    """
    # Calculate the distance of each symbol from each constellation point
    distances = np.abs(qam_symbols[:, None] - constellation[None, :])
    # Find the index of the minimum distance for each symbol
    indices = np.argmin(distances, axis=1)
    
    # Convert indices back to binary data
    binary_data = np.array([[(index >> bit) & 1 for bit in range(7, -1, -1)] for index in indices])
    binary_data = binary_data.flatten()
    
    # Convert binary data back to bipolar format (-1 for 0, and 1 for 1)
    ldpc_encoded_data_recovered = binary_data * 2 - 1
    
    # Remove padding by trimming the ldpc_encoded_data_recovered array to the original_data_length
    ldpc_encoded_data_recovered = ldpc_encoded_data_recovered[:original_data_length]
    
    return ldpc_encoded_data_recovered



def snr_to_noise_std_complex(snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    return noise_std


def channel_layer_numpy(qam_symbols, channel_type, snr_db, gain_mean=0, gain_std=10, K_factor=0):
    SNR_std = snr_to_noise_std_complex(snr_db)
    
    # Placeholder for the channel gain
    gain = None
    
    if channel_type == 2:  # Rayleigh
        gain = np.sqrt(np.random.normal(gain_mean, gain_std, qam_symbols.shape)**2 +
                       np.random.normal(gain_mean, gain_std, qam_symbols.shape)**2)
    elif channel_type == 3:  # Rician
        rayleigh_component = np.sqrt(np.random.normal(gain_mean, gain_std, qam_symbols.shape)**2 +
                                     np.random.normal(gain_mean, gain_std, qam_symbols.shape)**2)
        deterministic_component = np.sqrt(K_factor / (K_factor + 1)) * np.ones(qam_symbols.shape)
        gain = deterministic_component + np.sqrt(1 / (2 * (K_factor + 1))) * rayleigh_component

    if gain is not None:
        # Apply the channel gain to simulate transmission
        qam_symbols = qam_symbols * gain
    
    # Adding AWGN noise
    noise = np.random.normal(0, SNR_std, qam_symbols.shape) + 1j * np.random.normal(0, SNR_std, qam_symbols.shape)
    qam_symbols_noisy = qam_symbols + noise

    # Demodulation (assuming perfect CSI)
    if channel_type in [2, 3]:
        # Assuming perfect CSI, invert the channel effects
        qam_symbols_demodulated = qam_symbols_noisy / gain
    else:
        # For AWGN, no channel inversion necessary
        qam_symbols_demodulated = qam_symbols_noisy

    return qam_symbols_demodulated



def binary_to_integers(binary_data, bits_per_block=10):
    """
    Convert binary data to integers, assuming each block of bits represents one integer.
    
    :param binary_data: A numpy array of binary data (0s and 1s).
    :param bits_per_block: Number of bits per block used to represent each integer.
    :return: A numpy array of integers represented by the binary blocks.
    """
    # Reshape the binary data into blocks
    num_blocks = binary_data.shape[0] // bits_per_block
    binary_blocks = binary_data[:num_blocks * bits_per_block].reshape(-1, bits_per_block)
    
    # Convert each binary block to an integer
    integers = binary_blocks.dot(2**np.arange(bits_per_block)[::-1])
    
    return integers




# # Example usage
# n, dv, dc = 20, 2, 4  # LDPC parameters k=11, n =20
# H, G = make_ldpc(n, dv, dc, systematic=True, sparse=True)  # Generate LDPC code


# print('H:', H)
# print('G:', G)

# # Generate a random matrix of indices to represent the data to be transmitted

# len = 100
# indices_matrix = np.random.randint(0, 1024, size=(len, 8))

# # length = indices_matrix[0]
# # print(length)

# # Convert indices to binary
# binary_data = index_to_binary(indices_matrix, 10)

# ################## padding the 10 bits to 11 bits or 9 bits for ldpc

# print('indices_matrix', indices_matrix)
# print('binary_data', binary_data)

# print(type(binary_data))
# print(binary_data.shape)


# ################## cutting the binary data so that every 11 bits are processed by the encoder

# # Pad binary data for LDPC
# binary_data_padded = pad_binary_data_for_ldpc(binary_data)

# print('binary_data_padded:', binary_data_padded)

# # Encode the binary data using LDPC
# ldpc_encoded_data = ldpc_encode(binary_data_padded, H, G)


# print('ldpc_encoded_data:', ldpc_encoded_data)

# # Generate 256-QAM Constellation
# constellation_256qam = generate_256qam_constellation()

# # Modulate the LDPC encoded data using the 1024-QAM constellation
# qam_symbols = qam256_modulation(convert_bipolar_to_binary(ldpc_encoded_data), constellation_256qam)


# print("constellation", constellation_256qam)
# print('qam_symbols', qam_symbols)


# # Usage example
# channel_type = 2  # Rayleigh
# snr_db = 5  # Example SNR in dB

# # Simulate channel effects and demodulate assuming perfect CSI
# qam_symbols_demodulated = channel_layer_numpy(qam_symbols, channel_type, snr_db)

# print("Demodulated QAM symbols:", qam_symbols_demodulated)


# original_data_length = ldpc_encoded_data.shape[0]  # Length of the original LDPC-encoded data before any padding

# # Assuming qam_symbols is the result of the modulation process
# QAM_demodulated = qam256_demodulation_and_remove_padding(qam_symbols_demodulated, constellation_256qam, original_data_length)

# print("QAM_demodulated:", QAM_demodulated)


# # Perform decoding
# decoded_message = ldpc_decode(QAM_demodulated, H, G, snr=1000)
# print("Decoded message:", decoded_message)

# cut_binary_data = cut_binary_data_from_ldpc(decoded_message)
# print(cut_binary_data)


# print("Comparison report:", compare_and_report(binary_data, cut_binary_data))


# output = binary_to_integers(cut_binary_data)

# input_data = np.array(indices_matrix).flatten()
# print('input:', input_data)
# print('final output:', output)


# print("Comparison report:", compare_and_report(input_data, output))

#######################################################################################


# LDPC Code Generation
n, dv, dc = 20, 2, 4  # Parameters for LDPC
H, G = make_ldpc(n, dv, dc, systematic=True, sparse=True)

def transmission(input_indices_matrix, SNR_dB=5, gain_mean=0, gain_std=1, channel_type=2):
    # Step 1: Convert indices to binary
    binary_data = index_to_binary(input_indices_matrix, 10)
    
    # Step 2: LDPC Encoding

    binary_data_padded = pad_binary_data_for_ldpc(binary_data)
    ldpc_encoded_data = ldpc_encode(binary_data_padded, H, G)
    
    # Step 3: 256-QAM Modulation
    constellation_256qam = generate_256qam_constellation()
    qam_symbols = qam256_modulation(convert_bipolar_to_binary(ldpc_encoded_data), constellation_256qam)
    
    # Step 4: Channel Simulation
    qam_symbols_demodulated = channel_layer_numpy(qam_symbols, channel_type=2, snr_db=SNR_dB, gain_mean=gain_mean, gain_std=gain_std)
    
    # Step 5: 256-QAM Demodulation
    original_data_length = ldpc_encoded_data.shape[0]
    QAM_demodulated = qam256_demodulation_and_remove_padding(qam_symbols_demodulated, constellation_256qam, original_data_length)
    
    # Step 6: LDPC Decoding
    decoded_message = ldpc_decode(QAM_demodulated, H, G, snr=1000)
    cut_binary_data = cut_binary_data_from_ldpc(decoded_message)
    
    # Step 7: Binary to Integers Conversion
    output = binary_to_integers(cut_binary_data)
    return output


# # Example usage
# input_indices_matrix = np.random.randint(0, 1024, size=(100, 8))  # Sample input


# final_output = transmission(input_indices_matrix, SNR_dB=5, gain_mean=10, gain_std=1, channel_type=2)


# input_data = np.array(input_indices_matrix).flatten()
# print('input:', input_data)

# print("Final output:", final_output)

# print("Comparison report:", compare_and_report(input_data, final_output))

