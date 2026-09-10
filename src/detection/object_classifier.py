"""
Everyday Object Classifier Inference Module
Predicts everyday objects (ball, bottle, cup, phone, book, chair, laptop) from BGR image frames.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEFAULT_CLASSES = [
    'ball',
    'book',
    'bottle',
    'chair',
    'cup',
    'laptop',
    'phone'
]

class ObjectClassifier:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> Initializing ObjectClassifier on device: {self.device}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if model_path and os.path.exists(model_path):
            print(f">> Loading custom object model from '{model_path}'...")
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                self.categories = checkpoint.get('class_names', DEFAULT_CLASSES)
            else:
                state_dict = checkpoint
                self.categories = DEFAULT_CLASSES

            self.model = models.mobilenet_v3_large(weights=None)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, len(self.categories))
            self.model.load_state_dict(state_dict)
        else:
            print(">> Loading default pre-trained MobileNetV3 for Object Classification...")
            self.categories = DEFAULT_CLASSES
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, len(self.categories))

        self.model.to(self.device)
        self.model.eval()

    def predict_frame(self, cv2_frame):
        if cv2_frame is None:
            return "unknown", 0.0, {}

        rgb_frame = cv2_frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        prob_dict = {self.categories[i]: float(probabilities[i]) for i in range(len(self.categories))}
        top_idx = torch.argmax(probabilities).item()
        top_object = self.categories[top_idx]
        confidence = float(probabilities[top_idx])

        return top_object, confidence, prob_dict
