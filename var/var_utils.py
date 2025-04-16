import numpy as np
import PIL.Image as PImage
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import os
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

def multiclass_metrics(y_true, y_pred, num_classes, all_metrics=False):
    confusion = confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=np.arange(num_classes))  # Ensure correct labels
    
    # Per-class metrics
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP  # Column sum minus diagonal
    FN = np.sum(confusion, axis=1) - TP  # Row sum minus diagonal
    TN = np.sum(confusion) - (TP + FP + FN)  # Everything else

    accuracy = np.sum(TP) / np.sum(confusion) if np.sum(confusion) != 0 else 0
    if all_metrics:
        sensitivity = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)  # Recall
        specificity = np.divide(TN, (TN + FP), out=np.zeros_like(TN, dtype=float), where=(TN + FP) != 0)
        f1_or_dsc = np.divide(2 * TP, (2 * TP + FP + FN), out=np.zeros_like(TP, dtype=float), where=(2 * TP + FP + FN) != 0)
        jaccard = np.divide(TP, (TP + FP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FP + FN) != 0)
        miou = np.mean(jaccard)  # Mean IoU across classes

        return np.array([accuracy, sensitivity, specificity, f1_or_dsc, jaccard, miou])
    return accuracy

class CustomImageDataset(Dataset):
    def __init__(self, root, name="Training", transform=None):
        self.img_dir = os.path.join(root, name+"_Input")
        self.mask_dir = os.path.join(root, name+"_GroundTruth")

        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = PImage.open(img_path).convert("RGB")  # Convert to grayscale
        mask = PImage.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

def build_dataset(data_path: str):
    # build augmentations
    train_aug = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    val_aug = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # build dataset
    train_set = CustomImageDataset(root=data_path, name="Training", transform=train_aug)
    val_set = CustomImageDataset(root=data_path, name="Validation", transform=val_aug)
    
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}')
    
    return train_set, val_set