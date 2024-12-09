import torchaudio
import torch

# Function to extract features using wav2vec 2.0
def extract_features(waveform, model):
    # # Ensure the waveform is in the correct format for the model
    # waveform = waveform.to(model.device)
    
    # Extract the features from the waveform
    with torch.no_grad():
        features, _ = model(waveform)

    # You might need to perform pooling/aggregation over time steps depending on your FDSD calculation
    features = torch.mean(features, dim=1)
    
    return features.cpu()


path_original = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_original.wav'

path_clean = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_clean.wav'

path_noisy_random = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_noisy_random.wav'

path_noisy_sort = '/home/oem/Winfred/valle/egs/libritts/Results/1034_121119_000002_000001_nosiySort.wav'

# Load a pre-trained wav2vec 2.0 model
model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()

# Load your audio files and resample them to 16kHz (which is what wav2vec expects)
waveform_real, sample_rate_real = torchaudio.load(path_original)
waveform_synthesized, sample_rate_synthesized = torchaudio.load(path_clean)


# Define the expected sample rate for the wav2vec model
expected_sample_rate = 16000

# Resample audio to match the model's expected sample rate if necessary
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate_real, new_freq=expected_sample_rate)
waveform_real = resampler(waveform_real)

resampler = torchaudio.transforms.Resample(orig_freq=sample_rate_synthesized, new_freq=expected_sample_rate)
waveform_synthesized = resampler(waveform_synthesized)

# Extract features from the audio
features_real = extract_features(waveform_real, model)
features_synthesized = extract_features(waveform_synthesized, model)

# print(features_real)
# print(type(features_real))
# print(features_real.shape)

# print(features_synthesized)
# print(type(features_synthesized))
# print(features_synthesized.shape)


def calculate_fbsd(feature_real, feature_synthesized):
    """
    Calculate the Fréchet Deep Speech Distance (FDSD) between real and synthesized speech features.

    Args:
    feature_real (torch.Tensor): Real speech features tensor.
    feature_synthesized (torch.Tensor): Synthesized speech features tensor.

    Returns:
    float: The calculated FDSD value.
    """

    print("feature_real:", feature_real.shape)
    print("feature_synthesized:", feature_synthesized.shape)
   
    
    # Compute the mean and covariance of the real and synthesized features
    mu_real = torch.mean(feature_real, axis=0)
    mu_synthesized = torch.mean(feature_synthesized, axis=0)
    sigma_real = torch.cov(feature_real.T)
    sigma_synthesized = torch.cov(feature_synthesized.T)

    # Compute the sum of the product of covariances
    covmean = sigma_real @ sigma_synthesized
    
    # Compute eigenvalues and eigenvectors for the square root of the product of covariances
    eigvals, eigvecs = torch.linalg.eigh(covmean)
    
    # Take the square root of eigenvalues, ensuring they are positive by taking their absolute value
    sqrt_eigvals = torch.sqrt(torch.abs(eigvals))
    
    # Compute the square root of the product of covariances using eigenvalues and eigenvectors
    covmean_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T

    # Ensure that covmean_sqrt is Hermitian (symmetric if real-valued) and positive-semidefinite
    covmean_sqrt = (covmean_sqrt + covmean_sqrt.T) / 2

    # Compute the trace of sqrt of the product of covariances
    tr_covmean = torch.trace(covmean_sqrt)

    # Calculate the FDSD using the trace
    diff = mu_real - mu_synthesized

    print('diff:', diff)
    print('sigma_real:', sigma_real)
    print('sigma_synthesized:', sigma_synthesized)
    print('tr_covmean:', tr_covmean)


    fbsd = (diff @ diff) + torch.trace(sigma_real) + torch.trace(sigma_synthesized) - 2 * tr_covmean
    
    return fbsd.item()


# Calculate the FDSD
fbsd_value = calculate_fbsd(features_real, features_synthesized)

print(fbsd_value)

