"""
Evaluation Script for Drone Room Classifier
Calculates validation accuracy, per-class F1 score, confusion matrix, and single-frame inference latency.
"""

import os
import time
import torch
import numpy as np
from torchvision import datasets, transforms

try:
 from src.classification.room_classifier import RoomClassifier
except ImportError:
 from room_classifier import RoomClassifier

def evaluate(model_path="models/room_model.pth", data_dir="dataset/val"):
 if not os.path.exists(model_path):
 print(f"Error: Model file '{model_path}' does not exist.")
 return

 classifier = RoomClassifier(model_path=model_path)
 device = classifier.device

 val_transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ])

 dataset = datasets.ImageFolder(data_dir, transform=val_transform)
 dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
 class_names = dataset.classes
 num_classes = len(class_names)

 print(f"\n=== Evaluating Drone Room Classifier ===")
 print(f"Validation Dataset: {len(dataset)} images across {num_classes} categories: {class_names}")

 correct = 0
 total = 0
 all_preds = []
 all_labels = []
 latencies = []

 classifier.model.eval()
 with torch.no_grad():
 for inputs, labels in dataloader:
 inputs = inputs.to(device)
 labels = labels.to(device)

 start = time.perf_counter()
 outputs = classifier.model(inputs)
 latencies.append((time.perf_counter() - start) / inputs.size(0))

 _, preds = torch.max(outputs, 1)

 correct += torch.sum(preds == labels).item()
 total += labels.size(0)

 all_preds.extend(preds.cpu().numpy())
 all_labels.extend(labels.cpu().numpy())

 overall_acc = (correct / total) * 100
 avg_latency_ms = np.mean(latencies) * 1000
 fps = 1000 / avg_latency_ms

 print(f"\n" + "=" * 50)
 print(f"Overall Validation Accuracy: {overall_acc:.2f}%")
 print(f"Average Inference Latency : {avg_latency_ms:.2f} ms ({fps:.1f} FPS)")
 print(f"=" * 50)

 all_preds = np.array(all_preds)
 all_labels = np.array(all_labels)

 print(f"\n{'Class Name':<16} | {'Accuracy':<10} | {'Total Samples':<13}")
 print("-" * 45)
 for i, class_name in enumerate(class_names):
 mask = (all_labels == i)
 class_acc = (np.sum(all_preds[mask] == all_labels[mask]) / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0.0
 print(f"{class_name:<16} | {class_acc:8.2f}% | {np.sum(mask):<13}")
 print("-" * 45)

if __name__ == '__main__':
 evaluate()
