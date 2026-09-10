"""
Room Classifier Module for Tello Drone
Classifies image frames into indoor room categories using optimized MobileNetV3 / ResNet models.
Ultra-fast real-time frame inference for edge device deployment.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

ROOM_CATEGORIES = [
    'bathroom',
    'bedroom',
    'dining_room',
    'kitchen',
    'living_room'
]

class RoomClassifier:
    def __init__(self, model_path=None, device=None, default_arch="mobilenet_v3_large"):
        """
        Initializes PyTorch model for room classification.
        Supports MobileNetV3-Large, MobileNetV3-Small, ResNet18, and EfficientNet-B0 checkpoints.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> Initializing RoomClassifier on device: {self.device}")

        # Image Preprocessing Transform (ImageNet statistics)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.custom_model = False
        if model_path and os.path.exists(model_path):
            print(f">> Loading fine-tuned model checkpoint from '{model_path}'...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                arch = checkpoint.get('arch', default_arch)
                state_dict = checkpoint['state_dict']
                self.categories = checkpoint.get('class_names', ROOM_CATEGORIES)
                print(f"  [+] Loaded metadata: arch='{arch}', categories={self.categories}")
            else:
                state_dict = checkpoint
                arch = default_arch
                # Infer number of categories from weight matrix shape
                if 'classifier.3.weight' in state_dict:
                    num_classes = state_dict['classifier.3.weight'].shape[0]
                    arch = "mobilenet_v3_large"
                elif 'fc.weight' in state_dict:
                    num_classes = state_dict['fc.weight'].shape[0]
                    arch = "resnet18"
                elif 'classifier.1.weight' in state_dict:
                    num_classes = state_dict['classifier.1.weight'].shape[0]
                    arch = "efficientnet_b0"
                else:
                    num_classes = 5

                if num_classes == 5:
                    self.categories = ['bathroom', 'bedroom', 'dining_room', 'kitchen', 'living_room']
                elif num_classes == 6:
                    self.categories = ['bathroom', 'bedroom', 'dining_room', 'kitchen', 'living_room', 'office']
                else:
                    self.categories = ROOM_CATEGORIES[:num_classes]

            # Build model skeleton based on arch
            self.model = self._build_model_skeleton(arch, len(self.categories))
            self.model.load_state_dict(state_dict)
            self.custom_model = True

        else:
            print(f">> Loading pre-trained {default_arch} for room recognition...")
            self.categories = ROOM_CATEGORIES
            self.model = self._build_model_skeleton(default_arch, len(self.categories), pretrained=True)

        self.model.to(self.device)
        self.model.eval()

    def _build_model_skeleton(self, arch, num_classes, pretrained=False):
        if arch == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
        elif arch == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
        elif arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif arch == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)

        return model

    def predict_frame(self, cv2_frame):
        """
        Takes a BGR OpenCV image frame, runs inference, and returns predicted room & confidence.
        :param cv2_frame: numpy array (BGR image frame from drone camera)
        :return: (top_room_name, confidence_score, dict_of_all_probs)
        """
        if cv2_frame is None:
            return "unknown", 0.0, {}

        # Convert OpenCV BGR to PIL RGB
        rgb_frame = cv2_frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)

        # Transform and add batch dimension
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        prob_dict = {self.categories[i]: float(probabilities[i]) for i in range(len(self.categories))}
        
        top_idx = torch.argmax(probabilities).item()
        top_room = self.categories[top_idx]
        confidence = float(probabilities[top_idx])

        return top_room, confidence, prob_dict
