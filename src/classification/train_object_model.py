"""
PyTorch Everyday Object Classifier Training Script (MobileNetV3)
Trains MobileNetV3-Large to identify everyday objects (ball, bottle, cup, phone, book, chair, laptop).
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, models, transforms

OBJECT_CLASSES = [
 'ball',
 'bottle',
 'book',
 'chair',
 'cup',
 'laptop',
 'phone'
]

def get_model(num_classes=7, pretrained=True):
 weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
 model = models.mobilenet_v3_large(weights=weights)
 in_features = model.classifier[3].in_features
 model.classifier[3] = nn.Linear(in_features, num_classes)
 return model

def train_object_model(data_dir="dataset_objects", num_epochs=8, batch_size=16):
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print("=== Training Everyday Object Recognition Model (MobileNetV3) ===")
 print(f">> Target Classes : {OBJECT_CLASSES}")
 print(f">> Compute Device : {device}")

 # Data Augmentation Transforms
 data_transforms = {
 'train': transforms.Compose([
 transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
 transforms.RandomHorizontalFlip(),
 transforms.RandomRotation(degrees=15),
 transforms.ColorJitter(brightness=0.2, contrast=0.2),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ]),
 'val': transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ]),
 }

 train_path = os.path.join(data_dir, 'train')
 val_path = os.path.join(data_dir, 'val')

 if not os.path.exists(train_path):
 from download_object_dataset import download_and_setup
 download_and_setup(data_dir)

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
 print(f">> Dataset Split: {dataset_sizes['train']} train images, {dataset_sizes['val']} val images")

 model = get_model(num_classes=len(class_names), pretrained=True)
 model = model.to(device)

 criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

 # Freeze backbone initially for warmup
 for param in model.parameters():
 param.requires_grad = False
 for param in model.classifier[3].parameters():
 param.requires_grad = True

 optimizer = optim.AdamW(model.classifier[3].parameters(), lr=1e-3, weight_decay=1e-2)

 # 1. Warmup Epochs
 print("\n--- Phase 1: Classifier Head Warmup (2 Epochs) ---")
 for epoch in range(2):
 for phase in ['train', 'val']:
 model.train() if phase == 'train' else model.eval()
 running_loss, running_corrects = 0.0, 0

 for inputs, labels in dataloaders[phase]:
 inputs, labels = inputs.to(device), labels.to(device)
 optimizer.zero_grad()

 with torch.set_grad_enabled(phase == 'train'):
 outputs = model(inputs)
 _, preds = torch.max(outputs, 1)
 loss = criterion(outputs, labels)

 if phase == 'train':
 loss.backward()
 optimizer.step()

 running_loss += loss.item() * inputs.size(0)
 running_corrects += torch.sum(preds == labels.data)

 epoch_loss = running_loss / dataset_sizes[phase]
 epoch_acc = running_corrects.double() / dataset_sizes[phase]
 print(f" Warmup Epoch {epoch + 1}/2 [{phase.upper()}]: Loss {epoch_loss:.4f} Acc {epoch_acc:.4f}")

 # 2. Full Backbone Fine-Tuning
 print("\n--- Phase 2: Full Fine-Tuning ---")
 for param in model.parameters():
 param.requires_grad = True

 backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.3")]
 head_params = model.classifier[3].parameters()

 optimizer = optim.AdamW([
 {'params': backbone_params, 'lr': 1e-4},
 {'params': head_params, 'lr': 1e-3}
 ], weight_decay=1e-2)

 scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

 best_acc = 0.0
 best_model_wts = copy.deepcopy(model.state_dict())

 for epoch in range(num_epochs):
 print(f"\nFine-tune Epoch {epoch + 1}/{num_epochs}")
 for phase in ['train', 'val']:
 model.train() if phase == 'train' else model.eval()
 running_loss, running_corrects = 0.0, 0

 for inputs, labels in dataloaders[phase]:
 inputs, labels = inputs.to(device), labels.to(device)
 optimizer.zero_grad()

 with torch.set_grad_enabled(phase == 'train'):
 outputs = model(inputs)
 _, preds = torch.max(outputs, 1)
 loss = criterion(outputs, labels)

 if phase == 'train':
 loss.backward()
 optimizer.step()

 running_loss += loss.item() * inputs.size(0)
 running_corrects += torch.sum(preds == labels.data)

 epoch_loss = running_loss / dataset_sizes[phase]
 epoch_acc = running_corrects.double() / dataset_sizes[phase]
 print(f" [{phase.upper():5s}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

 if phase == 'val' and epoch_acc > best_acc:
 best_acc = epoch_acc
 best_model_wts = copy.deepcopy(model.state_dict())

 scheduler.step()

 print(f"\n==================================================")
 print(f"BEST OBJECT RECOGNITION VAL ACCURACY: {best_acc * 100:.2f}%")
 print(f"==================================================")

 # Save Checkpoint
 os.makedirs("models", exist_ok=True)
 save_path = os.path.join("models", "object_model.pth")
 checkpoint = {
 'arch': 'mobilenet_v3_large',
 'state_dict': best_model_wts,
 'class_names': class_names,
 'best_acc': float(best_acc)
 }
 torch.save(checkpoint, save_path)
 print(f"[SUCCESS] Object Recognition Model saved to '{save_path}'")

if __name__ == '__main__':
 train_object_model()
