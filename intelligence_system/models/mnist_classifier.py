"""
Professional MNIST Classifier
Saves/loads model, proper training loop
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import warnings
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class MNISTNet(nn.Module):
    """Simple but effective MNIST network"""
    
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MNISTClassifier:
    """Professional MNIST classifier with save/load"""
    
    def __init__(self, model_path: Path, hidden_size: int = 128, 
                 lr: float = 0.001, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = MNISTNet(hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Load if exists
        if model_path.exists():
            self.load()
            logger.info(f"✅ Loaded existing model from {model_path}")
        else:
            logger.info("🆕 Created new MNIST model")
        
        # Prepare data
        self.train_loader, self.test_loader = self._prepare_data()
    
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare MNIST dataloaders"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            './data', train=False, transform=transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        return train_loader, test_loader
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            # FIX P#5: Suppress incompletude coroutine warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*backward_with_incompletude.*")
                loss.backward()
            self.optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
        
        accuracy = 100. * correct / total
        return accuracy
    
    def evaluate(self) -> float:
        """Evaluate on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        accuracy = 100. * correct / total
        return accuracy
    
    def save(self):
        """Save model checkpoint"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.model_path)
            logger.info(f"💾 Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load(self):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"📂 Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make prediction on input tensor
        
        Args:
            x: Input tensor (batch_size, 1, 28, 28)
        
        Returns:
            Output logits (batch_size, 10)
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make classifier callable"""
        return self.predict(x)
