# model.py
import torch
import torch.nn as nn
import torchvision.models as models

def load_model(model_path, num_classes=3, device='cpu'):
    """Load a pre-trained ResNet model"""
    try:
        # Create model architecture
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        
        return model
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None
