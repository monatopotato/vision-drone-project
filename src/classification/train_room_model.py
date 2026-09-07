"""
Advanced Drone Room Classifier Training Script (MobileNetV3 95%+ Target)
Optimized with Mixup, RandomErasing, TrivialAugmentWide, and Resolution Scaling.
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, models, transforms
from PIL import Image

def mixup_data(x, y, alpha=0.2):
 """Applies Mixup augmentation to input batch."""
 if alpha > 0:
 lam = np.random.beta(alpha, alpha)
 else:
 lam = 1

 batch_size = x.size(0)
 index = torch.randperm(batch_size).to(x.device)

 mixed_x = lam * x + (1 - lam) * x[index]
 y_a, y_b = y, y[index]
 return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
 return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_model(arch="mobilenet_v3_large", num_classes=5, pretrained=True, dropout=0.2):
 """Instantiates MobileNetV3 with custom dropout and classification head."""
 weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
 model = models.mobilenet_v3_large(weights=weights)

 in_features = model.classifier[3].in_features
 # Add Dropout for strong regularization
 model.classifier = nn.Sequential(
 model.classifier[0], # Linear
 model.classifier[1], # Hardswish
 nn.Dropout(p=dropout),
 nn.Linear(in_features, num_classes)
 )
 return model

def freeze_backbone(model):
 for param in model.parameters():
 param.requires_grad = False
 for param in model.classifier.parameters():
 param.requires_grad = True

def unfreeze_all(model):
 for param in model.parameters():
 param.requires_grad = True

def train_model(data_dir="dataset", arch="mobilenet_v3_large", img_size=256, warmup_epochs=3, finetune_epochs=12, batch_size=16):
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print(f"=== Starting Advanced Drone Room Classifier Training (Target: 95%+) ===")
 print(f">> Backbone: {arch} | Input Resolution: {img_size}x{img_size}")
 print(f">> Device : {device}")

 # Optimized Image Transforms with TrivialAugment & RandomErasing
 data_transforms = {
 'train': transforms.Compose([
 transforms.Resize((img_size, img_size)),
 transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
 transforms.RandomHorizontalFlip(),
 transforms.RandomRotation(degrees=15),
 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
 transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
 ]),
 'val': transforms.Compose([
 transforms.Resize((img_size, img_size)),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ]),
 }

 # Load Image Datasets
 image_datasets = {
 x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
 for x in ['train', 'val']
 }
 dataloaders = {
 x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
 for x in ['train', 'val']
 }
 dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
 class_names = image_datasets['train'].classes

 print(f">> Loaded Classes ({len(class_names)}): {class_names}")
 print(f">> Train samples: {dataset_sizes['train']}, Val samples: {dataset_sizes['val']}")

 model = get_model(arch=arch, num_classes=len(class_names), pretrained=True, dropout=0.2)
 model = model.to(device)

 criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

 # Phase 1: Classifier Warmup
 print(f"\n--- Phase 1: Head Warmup ({warmup_epochs} epochs) ---")
 freeze_backbone(model)
 optimizer_head = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

 best_model_wts = copy.deepcopy(model.state_dict())
 best_acc = 0.0

 for epoch in range(warmup_epochs):
 print(f"Warmup Epoch {epoch + 1}/{warmup_epochs}")
 for phase in ['train', 'val']:
 model.train() if phase == 'train' else model.eval()
 running_loss, running_corrects = 0.0, 0

 for inputs, labels in dataloaders[phase]:
 inputs, labels = inputs.to(device), labels.to(device)
 optimizer_head.zero_grad()

 with torch.set_grad_enabled(phase == 'train'):
 outputs = model(inputs)
 _, preds = torch.max(outputs, 1)
 loss = criterion(outputs, labels)

 if phase == 'train':
 loss.backward()
 optimizer_head.step()

 running_loss += loss.item() * inputs.size(0)
 running_corrects += torch.sum(preds == labels.data)

 epoch_loss = running_loss / dataset_sizes[phase]
 epoch_acc = running_corrects.double() / dataset_sizes[phase]
 print(f" [{phase.upper():5s}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

 if phase == 'val' and epoch_acc > best_acc:
 best_acc = epoch_acc
 best_model_wts = copy.deepcopy(model.state_dict())

 # Phase 2: Full Fine-Tuning with Mixup
 print(f"\n--- Phase 2: Fine-Tuning with Mixup & Regularization ({finetune_epochs} epochs) ---")
 unfreeze_all(model)
 model.load_state_dict(best_model_wts)

 backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
 head_params = model.classifier.parameters()

 optimizer = optim.AdamW([
 {'params': backbone_params, 'lr': 5e-5},
 {'params': head_params, 'lr': 1e-3}
 ], weight_decay=2e-2)

 scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)

 start_time = time.time()
 for epoch in range(finetune_epochs):
 print(f"Epoch {epoch + 1}/{finetune_epochs} (Backbone LR: {optimizer.param_groups[0]['lr']:.6f})")
 for phase in ['train', 'val']:
 model.train() if phase == 'train' else model.eval()
 running_loss, running_corrects = 0.0, 0

 for inputs, labels in dataloaders[phase]:
 inputs, labels = inputs.to(device), labels.to(device)
 optimizer.zero_grad()

 with torch.set_grad_enabled(phase == 'train'):
 if phase == 'train':
 # Apply Mixup on training batch
 inputs_mix, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2)
 outputs = model(inputs_mix)
 loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
 loss.backward()
 optimizer.step()
 _, preds = torch.max(outputs, 1)
 else:
 outputs = model(inputs)
 loss = criterion(outputs, labels)
 _, preds = torch.max(outputs, 1)

 running_loss += loss.item() * inputs.size(0)
 running_corrects += torch.sum(preds == labels.data)

 epoch_loss = running_loss / dataset_sizes[phase]
 epoch_acc = running_corrects.double() / dataset_sizes[phase]
 print(f" [{phase.upper():5s}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

 if phase == 'val' and epoch_acc > best_acc:
 best_acc = epoch_acc
 best_model_wts = copy.deepcopy(model.state_dict())

 scheduler.step()

 total_time = time.time() - start_time
 print(f"\n==================================================")
 print(f"Training Completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
 print(f"BEST VALIDATION ACCURACY: {best_acc * 100:.2f}%")
 print(f"==================================================")

 # Save Checkpoint
 os.makedirs("models", exist_ok=True)
 model_save_path = os.path.join("models", "room_model.pth")

 checkpoint = {
 'arch': arch,
 'state_dict': best_model_wts,
 'class_names': class_names,
 'best_acc': float(best_acc)
 }
 torch.save(checkpoint, model_save_path)
 print(f"[SUCCESS] High-Accuracy MobileNetV3 model saved to '{model_save_path}'")

if __name__ == '__main__':
 train_model(arch="mobilenet_v3_large", img_size=256, warmup_epochs=3, finetune_epochs=12, batch_size=16)
