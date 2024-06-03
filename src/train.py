import torch 
import torch.nn as nn
import torchvision
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import argparse
import os


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        image = np.array(image)
        augmented = self.transform(image=image)
        return augmented['image']

def load_and_transform_data():
        
    train_transform = A.Compose(
    [
        A.Resize(128, 128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    )

    val_transform = A.Compose(
        [
            A.Resize(128, 128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    
    # Get the path to the input data from environment variable
    input_data_path = os.environ['AZURE_ML_INPUT_input_data']

    train_path = os.path.join(input_data_path, 'train')
    val_path = os.path.join(input_data_path, 'valid')
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=AlbumentationsTransform(train_transform))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=AlbumentationsTransform(val_transform))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    num_classes = len(train_dataset.classes)
    
    return train_dataloader, val_dataloader, num_classes
 
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", dest='learning_rate', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--num_epochs", dest='num_epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--pretrained", dest='pretrained', type=bool, default=False, help="Use pretrained model (True/False)")

    args = parser.parse_args()

    return args

def create_model(num_classes, pretrained):
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        
    accuracy = correct / len(train_loader.dataset)
    
    return model, running_loss / len(train_loader), accuracy

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    val_loss /= len(val_loader)

    accuracy = correct / len(val_loader.dataset)
    return  val_loss, accuracy

def run_training(model, train_dataloader, val_dataloader, 
                 criterion, optimizer, num_epochs, device):
    

    for epoch in range(num_epochs):
        model, train_loss, train_accuracy = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Save trained model
    model_path = os.path.join('outputs', "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main(args):
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    pretrained = args.pretrained

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, num_classes = load_and_transform_data()
    
    model = create_model(num_classes=num_classes, pretrained=pretrained)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model = model.to(device)

    run_training(model, train_dataloader, val_dataloader, 
                 criterion, optimizer,num_epochs, device)
    
if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)
    
    args = parse_args()
    
    main(args)
    
    print("*" * 60)
    print("\n\n")