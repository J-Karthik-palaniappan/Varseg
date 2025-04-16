
import torch
import os

import sys
sys.path.append("../")

from models.vqvae import VQVAE
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from vqvae.vqvae_utils import score, SingleImageDataset, display_results
import numpy as np
import matplotlib.pyplot as plt

def plot_hist_counts(filename="hist_counts.npz", save_path="histograms.png"):
    """Reads histogram counts from a file and plots them."""
    data = np.load(filename)
    hist_counts = data["hist_counts"]

    num_bins = hist_counts.shape[1]  # Get bin size from saved data

    # Create figure with subplots
    fig, axes = plt.subplots(hist_counts.shape[0] + 1, 1, figsize=(10, 12))  

    # Plot individual histograms
    for i, hist in enumerate(hist_counts):
        axes[i].bar(range(num_bins), hist, color='blue', alpha=0.7)
        axes[i].set_title(f"Histogram for array {i+1}")
        axes[i].set_xlim([0, num_bins])
        axes[i].set_ylabel("Count")

    # Plot combined histogram
    combined_hist = np.sum(hist_counts, axis=0)
    axes[-1].bar(range(num_bins), combined_hist, color='red', alpha=0.7)
    axes[-1].set_title("Combined Histogram (Sum of All)")
    axes[-1].set_xlim([0, num_bins])
    axes[-1].set_ylabel("Count")
    axes[-1].set_xlabel("Value (0-8192)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Save the plot
    plt.close()
    print(f"Histogram plot saved to {save_path}")

def get_weights(filename="hist_counts.npz", smoothing=0.1, eps=1e-6):
    data = np.load(filename)
    hist_counts = data["hist_counts"]

    combined_hist = np.sum(hist_counts, axis=0).astype(np.float32)

    # Avoid division by zero and suppress rare class over-weighting
    combined_hist[combined_hist == 0] = eps

    # Smoothed inverse frequency
    class_weights = 1.0 / (np.log(smoothing + combined_hist))

    # Normalize (optional)
    class_weights = class_weights / class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float32)

def get_weights2(filename="hist_counts.npz", smoothing=0.1, min_count_threshold=10):
    """
    Compute log-smoothed inverse class weights, with a threshold to avoid boosting rare classes.

    Parameters:
        filename (str): Path to .npz file with 'hist_counts' array.
        smoothing (float): Log smoothing factor.
        min_count_threshold (int): Minimum count below which weight is set to 0.
    
    Returns:
        torch.FloatTensor: Normalized class weights for CrossEntropy/Focal loss.
    """
    data = np.load(filename)
    hist_counts = data["hist_counts"]

    combined_hist = np.sum(hist_counts, axis=0).astype(np.float32)

    # Initialize weights
    class_weights = np.zeros_like(combined_hist, dtype=np.float32)

    # Apply weight only for sufficiently frequent classes
    mask = combined_hist >= min_count_threshold
    class_weights[mask] = 1.0 / (np.log(smoothing + combined_hist[mask]))

    # Normalize to sum to 1 (optional for stability)
    total = class_weights.sum()
    if total > 0:
        class_weights /= total

    return torch.tensor(class_weights, dtype=torch.float32)

def plot_hist_counts2(filename="hist_counts.npz", save_path="histograms.png"):
    """Reads histogram counts from a file, finds top-10 bins by count, 
    normalizes to percentages, and plots them in ascending order."""
    
    data = np.load(filename)
    hist_counts = data["hist_counts"]
    num_bins = hist_counts.shape[1]

    num_arrays = hist_counts.shape[0]
    fig, axes = plt.subplots(num_arrays + 1, 1, figsize=(10, 3 * (num_arrays + 1)))

    # Individual plots
    for i, hist in enumerate(hist_counts):
        total = hist.sum()
        percent_hist = (hist / total) * 100  # Normalize to percentage

        # Get top 10 indices and values
        top10_indices = np.argsort(percent_hist)[-10:]
        top10_values = percent_hist[top10_indices]

        # Sort top 10 by value for better visualization
        sorted_idx = np.argsort(top10_values)
        top10_indices = top10_indices[sorted_idx]
        top10_values = top10_values[sorted_idx]

        axes[i].bar(range(10), top10_values, tick_label=top10_indices)
        axes[i].set_title(f"Top 10 Values for Array {i+1}")
        axes[i].set_ylabel("Percentage (%)")

    # Combined histogram
    combined_hist = np.sum(hist_counts, axis=0)
    total_combined = combined_hist.sum()
    combined_percent = (combined_hist / total_combined) * 100

    top10_indices = np.argsort(combined_percent)[-10:]
    top10_values = combined_percent[top10_indices]
    sorted_idx = np.argsort(top10_values)
    top10_indices = top10_indices[sorted_idx]
    top10_values = top10_values[sorted_idx]

    axes[-1].bar(range(10), top10_values, tick_label=top10_indices, color='red')
    axes[-1].set_title("Top 10 Values for Combined Histogram")
    axes[-1].set_ylabel("Percentage (%)")
    axes[-1].set_xlabel("Value")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Top-10 histograms saved to {save_path}")

def test_vqvae_vec(vqvae,model_path, test_loader, filename = "hist_counts.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae.to(device)
    vqvae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vqvae.eval()
    
    num_bins = vqvae.vocab_size
    print("num vecs:",num_bins)
    hist_counts = np.zeros((10, num_bins))
    with torch.no_grad():
        for img in tqdm(test_loader, desc="Testing"):
            img = img.to(device)

            gt_idxBl = vqvae.img_to_idxBl(img)
            hist_counts += np.array([np.bincount(a.detach().cpu().flatten(), minlength=num_bins) for a in gt_idxBl])

            # break
    np.savez_compressed(filename, hist_counts=hist_counts)
    print(f"Histogram counts saved to {filename}")

# Call functions
if __name__ == "__main__":
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Change if running on multiple nodes
    os.environ["MASTER_PORT"] = "29502"  # Choose an available port
    os.environ["WORLD_SIZE"] = "1"  # Set to the number of processes (GPUs)
    os.environ["RANK"] = "0"  # Unique rank of the process

    dist.init_process_group(backend="nccl", init_method="env://")
    print("World Size:", dist.get_world_size())
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = SingleImageDataset("/media/viplab/DATADRIVE1/skin_lesion/ISIC2018/Test_GroundTruth/", task="mask", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Load one image at a time
    
    # vqvae = VQVAE(in_channels=3, vocab_size=1024, z_channels=64, ch=160, test_mode=False)
    # vqvae = VQVAE(in_channels=1, vocab_size=256, z_channels=64, ch=160, test_mode=False)
    # test_vqvae_vec(vqvae, "../vqvae/checkpoints/vqvae_best.pth", test_loader)
    # plot_hist_counts2()
    print(np.argsort(get_weights2()))