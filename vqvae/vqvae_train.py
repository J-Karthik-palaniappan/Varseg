import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("../")

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.vqvae import VQVAE
from torch.utils.data import DataLoader
from torchvision import transforms
from vqvae_utils import DiceLoss, PerceptualLoss, score, SingleImageDataset

import os
import torch.distributed as dist
os.environ["MASTER_ADDR"] = "127.0.0.1"  # Change if running on multiple nodes
os.environ["MASTER_PORT"] = "29500"  # Choose an available port
os.environ["WORLD_SIZE"] = "1"  # Set to the number of processes (GPUs)
os.environ["RANK"] = "0"  # Unique rank of the process

dist.init_process_group(backend="nccl", init_method="env://")
print("World Size:", dist.get_world_size())


# Training Function with Model Saving
def train_vqvae(vqvae, train_loader, val_loader, num_epochs=50, task="image", save_path="./checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae.to(device)
    vqvae.train()

    optimizer = optim.Adam(vqvae.parameters(), lr=1e-4)
    
    if task == "image":
        l1loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss(task="image")
    elif task == "mask":
        dice_loss = DiceLoss()
        perceptual_loss = PerceptualLoss(task="mask")
        
    writer = SummaryWriter()
    os.makedirs(save_path, exist_ok=True)  # Ensure checkpoint directory exists
    best_val_loss = float("inf")  # Track best validation loss

    for epoch in range(num_epochs):
        vqvae.train()
        total_loss, total_score = 0.0, 0.0

        for img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            img = img.to(device)
            optimizer.zero_grad()

            rec_img, _, vq_loss = vqvae(img)

            if task == "image": loss = l1loss(rec_img, img) + 0.5 * perceptual_loss(rec_img, img) + 0.1 * vq_loss
            else: loss = dice_loss(rec_img, img) + perceptual_loss(rec_img, img) + 0.1 * vq_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_score += score(rec_img.cpu().detach(), img.cpu().detach(), False)

        writer.add_scalar("Loss/Train", total_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/Train", total_score / len(train_loader), epoch)

        # Validation
        vqvae.eval()
        val_loss, val_score = 0.0, 0.0
        with torch.no_grad():
            for img in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                img = img.to(device)
                rec_img, _, _ = vqvae(img)

                if task == "image": loss = l1loss(rec_img, img) + 0.5 * perceptual_loss(rec_img, img)
                else: loss = dice_loss(rec_img, img) + perceptual_loss(rec_img, img)

                val_loss += loss.item()
                val_score += score(rec_img.cpu().detach(), img.cpu().detach(),False)

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_score / len(val_loader), epoch)

        # Save last model
        torch.save(vqvae.state_dict(), os.path.join(save_path, "vqvae_last.pth"))

        # Save best model (lowest validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vqvae.state_dict(), os.path.join(save_path, "vqvae_best.pth"))

        if epoch % 1 == 0:  # Log every 1 epochs
            writer.add_images("Images/Original", img[:5], epoch)
            writer.add_images("Images/Reconstructed", rec_img[:5], epoch)

        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {total_score/len(train_loader):.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val accuracy: {val_score/len(val_loader):.4f}")

    writer.close()
    print("Training Complete!")


# Main script
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = SingleImageDataset("/media/viplab/DATADRIVE1/skin_lesion/ISIC2018/Training_GroundTruth/", task="mask", transform=transform)
    val_dataset = SingleImageDataset("/media/viplab/DATADRIVE1/skin_lesion/ISIC2018/Validation_GroundTruth/", task="mask", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    #4096model : vocab_size=4096, z_channels=32, loss : dice + focal + vq
    #8192model : vocab_size=8192, z_channels=64, loss : dice + perceptual + 0.5*vq
    vqvae = VQVAE(in_channels=1, vocab_size=256, z_channels=64, ch=160, test_mode=False)
    # vqvae.load_state_dict(torch.load("checkpoints/img_new_best.pth"))
    train_vqvae(vqvae, train_loader, val_loader, num_epochs=50, task="mask", save_path="./checkpoints")