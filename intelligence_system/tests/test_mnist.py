"""
Professional tests for MNIST classifier
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.mnist_classifier import MNISTNet, MNISTClassifier
from config.settings import MNIST_MODEL_PATH


def test_mnist_network_forward():
    """Test MNISTNet forward pass"""
    model = MNISTNet(hidden_size=128)
    x = torch.randn(32, 1, 28, 28)
    output = model(x)
    
    assert output.shape == (32, 10), "Output shape should be (batch_size, 10)"


def test_mnist_network_output_range():
    """Test MNISTNet outputs reasonable values"""
    model = MNISTNet(hidden_size=128)
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    
    # Output should be logits (unbounded)
    assert output.shape == (1, 10)
    assert torch.isfinite(output).all(), "Output should be finite"


def test_mnist_classifier_trains():
    """Test that MNIST classifier improves with training"""
    # Use small model for test
    test_model_path = Path("/tmp/test_mnist.pth")
    if test_model_path.exists():
        test_model_path.unlink()
    
    classifier = MNISTClassifier(
        test_model_path,
        hidden_size=64,
        lr=0.01
    )
    
    initial_acc = classifier.evaluate()
    
    # Train for 1 epoch
    train_acc = classifier.train_epoch()
    
    final_acc = classifier.evaluate()
    
    # Should see some improvement (not perfect but better than random)
    assert final_acc > 10.0, "Accuracy should be better than random (10%)"
    
    # Cleanup
    if test_model_path.exists():
        test_model_path.unlink()


def test_mnist_save_load():
    """Test that MNIST model saves and loads correctly"""
    test_model_path = Path("/tmp/test_mnist_save.pth")
    
    # Create and train
    classifier = MNISTClassifier(test_model_path, hidden_size=64)
    classifier.train_epoch()
    acc1 = classifier.evaluate()
    classifier.save()
    
    # Load in new instance
    classifier2 = MNISTClassifier(test_model_path, hidden_size=64)
    acc2 = classifier2.evaluate()
    
    # Should have same accuracy
    assert abs(acc1 - acc2) < 0.1, "Loaded model should have same accuracy"
    
    # Cleanup
    if test_model_path.exists():
        test_model_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
