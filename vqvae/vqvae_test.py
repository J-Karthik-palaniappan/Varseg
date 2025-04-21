import torch
import os

import sys
sys.path.append("../")

from models.vqvae import VQVAE
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from vqvae_utils import score, SingleImageDataset, display_results
import numpy as np

import torch.distributed as dist
os.environ["MASTER_ADDR"] = "127.0.0.1"  # Change if running on multiple nodes
os.environ["MASTER_PORT"] = "29502"  # Choose an available port
os.environ["WORLD_SIZE"] = "1"  # Set to the number of processes (GPUs)
os.environ["RANK"] = "0"  # Unique rank of the process

dist.init_process_group(backend="nccl", init_method="env://")
print("World Size:", dist.get_world_size())

# Load the best model and evaluate a subset of test images
def test_vqvae(vqvae, model_path, test_loader, indices=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae.to(device)
    vqvae.load_state_dict(torch.load(model_path, map_location=device))
    vqvae.eval()

    metrics = None
    selected_images = []
    
    with torch.no_grad():
        for i, img in tqdm(enumerate(test_loader), desc="Testing"):
            if indices and i not in indices: continue  # Skip images not in the selected indices

            img = img.to(device)
            rec_img, _, _ = vqvae(img)
            
            try:
                if metrics is not None: metrics += score(rec_img.detach().cpu().numpy(), img.detach().cpu().numpy(), True)
                else: metrics = score(rec_img.detach().cpu().numpy(), img.detach().cpu().numpy(), True)
            except Exception as e: print(i, ":", e)

            if indices:
                original = transforms.ToPILImage()(img.squeeze(0).cpu())
                reconstructed = transforms.ToPILImage()(rec_img.squeeze(0).cpu())
                selected_images.append((original, reconstructed))
                if len(selected_images) == len(indices): break
    
    if indices: 
        print(metrics/len(indices))
        display_results(selected_images, save_path="outs/3.png")
    else: 
        print(metrics/len(test_loader))

# Main script
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = SingleImageDataset("/media/viplab/DATADRIVE1/skin_lesion/ISIC2018/Test_Input/", task="image", transform=transform)
    # test_dataset = SingleImageDataset("/media/viplab/DATADRIVE1/skin_lesion/ISIC2018/Test_GroundTruth/", task="mask", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Load one image at a time

    # Set indices of test images to evaluate (or None for sequential images)
    test_indices = [0, 5, 10, 18, 20, 25, 30, 40, 57, 58]
    vqvae = VQVAE(in_channels=3, vocab_size=1024, z_channels=64, ch=160, test_mode=False)
    # vqvae = VQVAE(in_channels=1, vocab_size=256, z_channels=64, ch=160, test_mode=False)
    test_vqvae(vqvae, "./checkpoints/img_best.pth", test_loader, indices=test_indices)