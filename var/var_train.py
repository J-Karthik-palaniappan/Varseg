import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("../")

from models import build_vae_var, build_vae_var2, build_vae_var3
from var_utils import multiclass_metrics, build_dataset
from var_test import test_var
from vqvector.plots import get_weights2

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

import os
import torch.distributed as dist
os.environ["MASTER_ADDR"] = "127.0.0.1"  # Change if running on multiple nodes
os.environ["MASTER_PORT"] = "29500"  # Choose an available port
os.environ["WORLD_SIZE"] = "1"  # Set to the number of processes (GPUs)
os.environ["RANK"] = "0"  # Unique rank of the process

dist.init_process_group(backend="nccl", init_method="env://")
print("World Size:", dist.get_world_size())


# Training Function with Model Saving
def train_var(var, vae_mask, train_loader, val_loader, num_epochs=50, save_path="./checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    var.to(device)
    var.train()

    optimizer = optim.Adam(var.parameters(), lr=1e-4)
    class_weights = get_weights2("../vqvector/hist_counts.npz").to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weights)
        
    writer = SummaryWriter()
    os.makedirs(save_path, exist_ok=True)  # Ensure checkpoint directory exists
    best_val_loss = float("inf")  # Track best validation loss
    V = vae_mask.vocab_size # Classification vocab size

    for epoch in range(num_epochs):
        var.train()
        total_loss, total_score = 0.0, 0.0

        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            img = img.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()

            with torch.no_grad():  # No gradients for vae_mask
                gt_idxBl = vae_mask.img_to_idxBl(mask)
                gt_BL = torch.cat(gt_idxBl, dim=1)
                x_BLCv_wo_first_l = vae_mask.quantize.idxBl_to_var_input(gt_idxBl)

            logits_BLV = var(x_BLCv_wo_first_l, img, teacher_force=torch.rand(1).item() < 0.97**epoch)
            loss = criterion(logits_BLV.view(-1, V), gt_BL.view(-1))

            loss.backward()
            optimizer.step()

            # # checking gradient flow EDITED
            # for name, param in var.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is None:
            #             # print(f"[NO GRAD] {name}")
            #             pass
            #         else:
            #             grad_mean = param.grad.abs().mean().item()
            #             grad_max = param.grad.abs().max().item()
            #             if grad_mean < 1e-7:
            #                 print(f"[TINY GRAD] {name} | mean: {grad_mean:.2e}, max: {grad_max:.2e}")

            total_loss += loss.item()
            total_score += multiclass_metrics(gt_BL.detach().cpu(), logits_BLV.argmax(dim=-1).detach().cpu(), num_classes=V, all_metrics=False)

        writer.add_scalar("Loss/Train", total_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/Train", total_score / len(train_loader), epoch)

        # Validation
        var.eval()
        val_loss, val_score = 0.0, 0.0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                img = img.to(device)
                mask = mask.to(device)

                gt_idxBl = vae_mask.img_to_idxBl(mask)
                gt_BL = torch.cat(gt_idxBl, dim=1)
                x_BLCv_wo_first_l = vae_mask.quantize.idxBl_to_var_input(gt_idxBl)
                logits_BLV = var(x_BLCv_wo_first_l, img, teacher_force=False)

                loss = criterion(logits_BLV.view(-1, V), gt_BL.view(-1))

                val_loss += loss.item()
                val_score += multiclass_metrics(gt_BL.detach().cpu(), logits_BLV.argmax(dim=-1).detach().cpu(), num_classes=V, all_metrics=False)

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_score / len(val_loader), epoch)

        # Save last model
        torch.save(var.state_dict(), os.path.join(save_path, "var_last.pth"))

        # Save best model (lowest validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(var.state_dict(), os.path.join(save_path, "var_best.pth"))

        if epoch % 1 == 0:  # Log every 1 epochs
            test_var(var, val_loader, indices=[0,1,2,3,4])

        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {total_score/len(train_loader):.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val accuracy: {val_score/len(val_loader):.4f}")

    writer.close()
    print("Training Complete!")


# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = build_dataset("/media/viplab/DATADRIVE1/skin_lesion/ISIC2018/")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = 2
    if model == 1:
        vae_img, vae_mask, var = build_vae_var(
            V=256, Vimg=1024, Cvae=64, ch=160, share_quant_resi=4,
            device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            depth=16, shared_aln=False,
            flash_if_available=True, fused_if_available=True,
            init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        )

        vae_img_ckpt = "../vqvae/checkpoints/img_best.pth"
        vae_mask_ckpt = "../vqvae/checkpoints/mask_best.pth"
        vae_mask.load_state_dict(torch.load(vae_mask_ckpt, map_location=device), strict=True)
        vae_img.load_state_dict(torch.load(vae_img_ckpt, map_location=device), strict=True)
        var.load_state_dict(torch.load("../var/checkpoints/var_last_save.pth", map_location=device))
    
        # Ensure vae_img and vae_mask are NOT trained
        vae_img.to(device).eval()
        vae_mask.to(device).eval()
        for param in vae_img.parameters():
            param.requires_grad = False
        for param in vae_mask.parameters():
            param.requires_grad = False
    elif model==2:
        vae_mask, var = build_vae_var2(
            V=256, Cvae=64, ch=160, share_quant_resi=4,
            device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            depth=16, shared_aln=False,
            flash_if_available=True, fused_if_available=True,
            init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        )

        vae_mask_ckpt = "../vqvae/checkpoints/mask_best.pth"
        vae_mask.load_state_dict(torch.load(vae_mask_ckpt, map_location=device), strict=True)
        # var.load_state_dict(torch.load("../var/checkpoints/var_last.pth", map_location=device))
    
        # Ensure vae_img and vae_mask are NOT trained
        vae_mask.to(device).eval()
        for param in vae_mask.parameters():
            param.requires_grad = False
    else:
        vae_img, vae_mask, var = build_vae_var3(
            V=256, Vimg=1024, Cvae=64, ch=160, share_quant_resi=4,
            device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            depth=16, shared_aln=False,
            flash_if_available=True, fused_if_available=True,
            init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        )

        vae_img_ckpt = "../vqvae/checkpoints/img_best.pth"
        vae_mask_ckpt = "../vqvae/checkpoints/mask_best.pth"
        vae_mask.load_state_dict(torch.load(vae_mask_ckpt, map_location=device), strict=True)
        vae_img.load_state_dict(torch.load(vae_img_ckpt, map_location=device), strict=True)
        # var.load_state_dict(torch.load("../var/checkpoints/var_last_save.pth", map_location=device))
    
        # Ensure vae_img and vae_mask are NOT trained
        vae_img.to(device).eval()
        vae_mask.to(device).eval()
        for param in vae_img.parameters():
            param.requires_grad = False
        for param in vae_mask.parameters():
            param.requires_grad = False
    train_var(var, vae_mask, train_loader, val_loader, num_epochs=50, save_path="./checkpoints")