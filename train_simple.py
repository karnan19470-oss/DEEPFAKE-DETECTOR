"""
train_simple.py — Simple training script for your model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
import numpy as np
from predict_face import build_model, DEVICE

class SimpleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[idx]

def train_with_sample_data():
    """Train with sample data if you have it"""
    
    # Create dummy data for testing (replace with your real data)
    print("Creating dummy training data...")
    
    # You need to replace this with your actual dataset
    # Structure: data/real/ and data/fake/
    
    data_dir = "deepfake_dataset"
    
    if not os.path.exists(data_dir):
        print(f"\n❌ Dataset folder '{data_dir}' not found!")
        print("\nPlease create this structure:")
        print(f"  {data_dir}/")
        print("    real/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    fake/")
        print("      image1.jpg")
        print("      image2.jpg")
        return
    
    # Load images
    real_paths = [os.path.join(data_dir, 'real', f) for f in os.listdir(os.path.join(data_dir, 'real')) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    fake_paths = [os.path.join(data_dir, 'fake', f) for f in os.listdir(os.path.join(data_dir, 'fake')) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(real_paths) == 0 or len(fake_paths) == 0:
        print("Need both real and fake images!")
        return
    
    print(f"Found {len(real_paths)} real and {len(fake_paths)} fake images")
    
    # Prepare data
    all_paths = real_paths + fake_paths
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths)
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = SimpleDataset(all_paths, all_labels, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model
    model = build_model(pretrained=True)
    model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training
    print("\nStarting training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}")
        
        accuracy = 100. * correct / len(dataset)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={accuracy:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'deepfake_resnet18_best.pth')
    print("\n✅ Model saved as 'deepfake_resnet18_best.pth'")

if __name__ == "__main__":
    train_with_sample_data()