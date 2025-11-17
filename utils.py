# utils.py - OpenCV-free implementation
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def preprocess_image(image):
    """Preprocess image using PIL only"""
    if isinstance(image, str):
        image = Image.open(image)
    
    # Ensure image is RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def predict_class(model, image, class_names, device):
    """Predict class for an image"""
    try:
        if model is None:
            # Mock response for demo
            return "women", 85.0, np.array([0.1, 0.2, 0.7])
        
        input_tensor = preprocess_image(image).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return class_names[predicted.item()], confidence.item() * 100, probabilities.cpu().numpy()[0]
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown", 0.0, np.zeros(len(class_names))

def compute_similarity(model, image1, image2, device):
    """Compute similarity between two images"""
    try:
        if model is None:
            # Mock similarity for demo
            return 75.0
        
        # Extract features from both images
        feat1 = extract_features(model, image1, device)
        feat2 = extract_features(model, image2, device)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(feat1, feat2)
        return similarity.item() * 100
    
    except Exception as e:
        print(f"Similarity computation error: {e}")
        return 0.0

def extract_features(model, image, device):
    """Extract features from image using model"""
    input_tensor = preprocess_image(image).to(device)
    
    model.eval()
    with torch.no_grad():
        # Assuming model has a feature extraction method
        # Adjust this based on your actual model architecture
        features = model.features(input_tensor)
        features = features.view(features.size(0), -1)
    
    return features

def is_dupe(similarity, threshold):
    """Determine if images are dupes based on similarity threshold"""
    return similarity >= threshold
