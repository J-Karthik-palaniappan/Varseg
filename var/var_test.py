import torch
import sys
sys.path.append("../")

from models import build_vae_var,  build_vae_var2, build_vae_var3

from vqvae.vqvae_utils import display_results
from var_utils import build_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def test_var(var, test_loader, indices=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_images = []
    
    with torch.no_grad():
        for i, (img, mask) in tqdm(enumerate(test_loader), desc="Testing"):
            if indices and i not in indices: continue  # Skip images not in the selected indices
            
            img = img.to(device)
            rec_img = var.autoregressive_infer_cfg(img, more_smooth=False)
                
            if indices:
                original = transforms.ToPILImage()(img[0].cpu())
                reconstructed = transforms.ToPILImage()(rec_img[0].cpu())
                selected_images.append((original, reconstructed))
                if len(selected_images) == len(indices): break

    # Display the images in an 8x2 grid
    display_results(selected_images, save_path="outs/222.png")
    if indices: return selected_images

# Main script
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset, val_dataset = build_dataset("/media/viplab/DATADRIVE1/skin_lesion/ISIC2018/")
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # Load one image at a time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = 2
    if model == 1:
        vae_img, vae_mask, var = build_vae_var(
            V=256, Vimg=1024, Cvae=64, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            depth=16, shared_aln=False,
        )
        vae_img.load_state_dict(torch.load("../vqvae/checkpoints/img_best.pth", map_location=device))
        vae_mask.load_state_dict(torch.load("../vqvae/checkpoints/mask_best.pth", map_location=device))
        # var.load_state_dict(torch.load("./checkpoints/var_last.pth", map_location=device))
    elif model==2:
        vae_mask, var = build_vae_var2(
            V=256, Cvae=64, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            depth=8, shared_aln=False,
        )
        vae_mask.load_state_dict(torch.load("../vqvae/checkpoints/mask_best.pth", map_location=device))
        # var.load_state_dict(torch.load("./checkpoints/many_ep.pth", map_location=device))
    else:
        vae_img, vae_mask, var = build_vae_var3(
            V=256, Vimg=1024, Cvae=64, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            depth=16, shared_aln=False,
        )
        vae_img.load_state_dict(torch.load("../vqvae/checkpoints/img_best.pth", map_location=device))
        vae_mask.load_state_dict(torch.load("../vqvae/checkpoints/mask_best.pth", map_location=device))
        # var.load_state_dict(torch.load("./checkpoints/var_last.pth", map_location=device))

    test_indices = [0, 5, 10, 18, 20, 25, 30, 40]  # Example indices (max 8)
    test_var(var, test_loader, indices=test_indices)