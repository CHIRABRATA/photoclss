import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from PIL import Image

# Load a pre-trained feature extractor (ResNet50 backbone for feature vectors)
@torch.no_grad()
def get_reid_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the classification head to output raw 2048-D feature vectors
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


def get_reid_extractor():
    """Backward-compatible API expected by utils.detector."""
    return get_reid_model()

# Image preprocessing transform for ReID input
reid_transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_body_embedding(pil_image, bbox, model):
    """
    Crops the body box [x1, y1, x2, y2] from the image and extracts a normalized 2048-D embedding.
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Crop body bounding box
    crop = pil_image.crop((x1, y1, x2, y2))
    if crop.size[0] == 0 or crop.size[1] == 0:
        return None
        
    tensor_img = reid_transform(crop).unsqueeze(0)
    
    with torch.no_grad():
        feature = model(tensor_img).squeeze().numpy()
        # L2 normalization
        norm_feature = feature / np.linalg.norm(feature)
        
    return norm_feature