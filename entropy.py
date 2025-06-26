import pandas as pd
import numpy as np
import npeet_plus as ne
from npeet.entropy_estimators import mi  # mutual-information

def discrete_entropy(x, nbins=16):
    """Plug-in (ML) entropy of a 1-D sample, in bits."""
    hist, _ = np.histogram(x, bins=nbins, range=(0,1), density=False)
    p = hist[hist > 0] / len(x)
    return -np.sum(p * np.log2(p))

# Iterate through stages 0 to 7
for stage_number in range(9):
    print(f"\n=== Processing Stage {stage_number} ===")
    
    # Construct Excel file path dynamically
    excel_path = f"/home/oem/Winfred/Amphion/VALLE/Data_VALLE/output/CodecStage/Stage{stage_number}/merged_all_stage_table.xlsx"
    print(excel_path)

    # Read the Excel file
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_path}")
        continue

    # Define the types of information to process
    info_types = ['CER', 'WER', 'similarity']
    
    for info_type in info_types:
        print(f"\n--- Processing {info_type} ---")
        
        # Extract columns as vectors
        loss_full = np.array(df[f'{info_type}_all'], dtype=np.float32)  # e.g., CER_all, WER_all, similarity_all
        loss_masked = np.array(df[f'{info_type}_Stage{stage_number}'], dtype=np.float32)  # e.g., CER_StageX, WER_StageX, similarity_StageX

        # Apply transformation for similarity
        if info_type == 'similarity':
            loss_full = 1.0 - loss_full
            loss_masked = 1.0 - loss_masked

        print("loss_full:", loss_full)
        print("loss_masked:", loss_masked)

        # Print data statistics for debugging
        print(f"{info_type}_all: {len(loss_full)} samples, mean = {np.mean(loss_full):.4f}, std = {np.std(loss_full):.4f}, unique = {len(np.unique(loss_full))}")
        print(f"{info_type}_Stage{stage_number}: {len(loss_masked)} samples, mean = {np.mean(loss_masked):.4f}, std = {np.std(loss_masked):.4f}, unique = {len(np.unique(loss_masked))}")


        # from npeet.entropy_estimators import entropy                         # H(·)

        # print("loss_masked:", loss_masked)

        # print("loss_full:", loss_full)
        
        # # 2. k-NN entropy estimates (reshape → (n,1))
        # H_Y = entropy(loss_masked.reshape(-1,1), k=20, base=2)         # H(Y)
        # H_Y_given = entropy(loss_full.reshape(-1,1), k=20, base=2)     # H(Y|Xi)

        # print(f"H_Y: {H_Y:.3f}")
        # print(f"H_Y_given: {H_Y_given:.3f}")

        # I_bits = H_Y - H_Y_given                                      # MI in *bits*
        # print(f"I(X_i;Y) ≈ {I_bits:.3f} bits  (k-NN entropy)")


        """
        Compute mutual information I(X_i; Y) between stream presence/absence (X_i) and performance metric (Y).
        
        Parameters:
        - loss_masked: Array of performance metrics (e.g., CER) when Stream D_i is removed (~470 samples).
        - loss_full: Array of performance metrics when all streams are used (~470 samples).
        - k: Number of neighbors for k-NN entropy estimation (default=5).
        - eps: Jittering noise scale to avoid duplicates (default=1e-4).
        - seed: Random seed for reproducibility (default=0).
        - bootstrap: Whether to use bootstrapping for stability (default=False).
        - n_boot: Number of bootstrap iterations (default=100).
        
        Returns:
        - I_bits: Mutual information in bits.
        """
        from npeet.entropy_estimators import mi

        eps=1e-4
        # Set random seed
        rng = np.random.default_rng(seed=0)
        
        # # Normalize to [0, 1]
        # loss_masked_norm = (loss_masked - loss_masked.min()) / (loss_masked.max() - loss_masked.min() + 1e-10)
        # loss_full_norm = (loss_full - loss_full.min()) / (loss_full.max() - loss_full.min() + 1e-10)
        
        # # Add jittering
        # loss_masked_jitter = loss_masked_norm + rng.normal(0, eps, loss_masked.shape)
        # loss_full_jitter = loss_full_norm + rng.normal(0, eps, loss_full.shape)
        
        # # Create X_i: Binary (0 = D_i absent, 1 = D_i present)
        # X_i = np.concatenate([np.zeros(len(loss_masked)), np.ones(len(loss_full))])
        
        # # Create Y: Performance metric
        # Y = np.concatenate([loss_masked_jitter, loss_full_jitter])

        
        # Create X_i: Binary (0 = D_i absent, 1 = D_i present)
        X_i = np.concatenate([np.zeros(len(loss_masked)), np.ones(len(loss_full))])
        
        # Create Y: Performance metric
        Y = np.concatenate([loss_masked, loss_full])

        Y_norm = (Y - Y.min()) / (Y.max() - Y.min() + 1e-10)
        Y_jitter = Y_norm + rng.normal(0, eps, Y_norm.shape)

        Y = Y_jitter

        bootstrap=True
        n_boot=100
        
        if bootstrap:
            # Bootstrap for stability
            I_bits_boot = []
            for _ in range(n_boot):
                idx = rng.choice(len(Y), size=len(Y), replace=True)
                I_bits = mi(X_i[idx].reshape(-1,1), Y[idx].reshape(-1,1), k=20, base=2)
                I_bits_boot.append(max(0, I_bits))
            I_bits = np.mean(I_bits_boot)
        else:
            # Direct MI
            I_bits = mi(X_i.reshape(-1,1), Y.reshape(-1,1), k=5, base=2)
            I_bits = max(0, I_bits)
        
        print(f"I(X_i;Y) ≈ {I_bits:.3f} bits (k-NN direct MI, bootstrapped)")
