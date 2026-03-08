"""
AlexNet Implementation in PyTorch

Paper: ImageNet Classification with Deep Convolutional Neural Networks (2012)
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

Author: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet architecture for image classification.
    
    Key features:
    - 5 convolutional layers
    - 3 fully connected layers
    - ReLU activation
    - Dropout for regularization
    - Local Response Normalization (LRN)
    - Max pooling
    """
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv1: 224x224x3 -> 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55x55 -> 27x27
            
            # Conv2: 27x27x96 -> 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27 -> 13x13
            
            # Conv3: 13x13x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13 -> 6x6
        )
        
        # Classifier layers
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def alexnet(pretrained: bool = False, num_classes: int = 1000) -> AlexNet:
    """
    Create an AlexNet model.
    
    Args:
        pretrained: If True, return model pretrained on ImageNet (not implemented)
        num_classes: Number of output classes
        
    Returns:
        AlexNet model
    """
    model = AlexNet(num_classes=num_classes)
    return model


if __name__ == "__main__":
    # Test the model
    model = alexnet(num_classes=1000)
    
    # Print model architecture
    print("=" * 50)
    print("AlexNet Architecture")
    print("=" * 50)
    print(model)
    print("=" * 50)
    print(f"Total parameters: {model.count_parameters():,}")
    print("=" * 50)
    
    # Test with random input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (first sample, first 10 classes): {output[0, :10]}")