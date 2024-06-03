import os
import json
import torch
import torchvision
import torch.nn as nn
import base64
from PIL import Image 
import io
import numpy as np 
import albumentations as A
from albumentations.pytorch import ToTensorV2

def init():
    global model
    # Get the path to the model file from the environment variable
    model_path = os.getenv('AZUREML_MODEL_DIR')
    full_model_path = os.path.join(model_path, 'model.pth')
    
    model = torchvision.models.resnet18(pretrained=False)
    
    # Hardcoded number of classes
    model.fc = nn.Linear(model.fc.in_features, 53)

    model.load_state_dict(torch.load(full_model_path, map_location=torch.device('cpu')))
    model.eval()

def run(raw_data):
    val_transform = A.Compose(
    [
        A.Resize(128, 128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
    try:
        if os.path.isfile(raw_data):
            # Handle file input, for test inside scripts
            with open(raw_data, 'r') as f:
                data = json.load(f)
        else:
            # Handle JSON string input, for Azure ML deployment website 
            data = json.loads(raw_data)
            
        encoded_image = data["inputs"]
        image_bytes = base64.b64decode(encoded_image)

        image = Image.open(io.BytesIO(image_bytes))
        inputs = val_transform(image=np.array(image))['image']
        inputs = inputs.unsqueeze(0)

        outputs = model(inputs)
        pred = torch.argmax(outputs, dim=-1)

        return pred.detach().numpy().tolist()
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})