"""
CNN Example - Common Convolutional Neural Network Patterns in PyTorch

This file demonstrates common CNN building blocks and patterns used in modern
deep learning. It provides a flexible, configurable CNN class that can be
easily adapted for various image classification tasks.

Key Concepts Covered:
- Convolutional layers (Conv2d)
- Pooling layers (MaxPool2d, AvgPool2d)
- Normalization (BatchNorm2d)
- Regularization (Dropout)
- Activation functions (ReLU)
- Skip connections (ResNet-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A reusable convolutional block that combines common operations.
    
    This block demonstrates the typical pattern of:
    Conv2d -> BatchNorm -> Activation -> (Optional: Pooling)
    
    Using batch normalization helps stabilize training and allows
    higher learning rates. It's now standard practice in most CNNs.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
        activation: str = "relu",
        use_pool: bool = False,
        pool_size: int = 2
    ):
        """
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale)
            out_channels: Number of output channels (number of filters)
            kernel_size: Size of the convolutional kernel (default 3x3)
            stride: Step size for the convolution (default 1)
            padding: Zero-padding added to maintain spatial dimensions
            use_batchnorm: Whether to include BatchNorm2d (recommended)
            activation: Activation function type ('relu', 'leaky_relu', 'elu')
            use_pool: Whether to include MaxPool2d at the end
            pool_size: Size of the pooling kernel
        """
        super(ConvBlock, self).__init__()
        
        # Convolutional layer: extracts features from input
        # in_channels -> out_channels feature maps
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm  # BatchNorm includes bias, so we can skip it here
        )
        
        # Batch Normalization: normalizes activations to stabilize training
        # This helps prevent "internal covariate shift"
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        # Activation function: introduces non-linearity
        # Without non-linearity, multiple layers would collapse to a single linear transformation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # Max Pooling: reduces spatial dimensions, provides translation invariance
        # Also reduces computational cost and helps prevent overfitting
        self.pool = nn.MaxPool2d(kernel_size=pool_size) if use_pool else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class SimpleCNN(nn.Module):
    """
    A flexible CNN for image classification.
    
    This architecture demonstrates common CNN patterns:
    1. Stacked convolutional layers with increasing channels
    2. Spatial downsampling via pooling
    3. Global average pooling (alternative to flattening)
    4. Dropout for regularization
    5. Fully connected classifier head
    
    The architecture is configurable, allowing you to adjust:
    - Number of convolutional layers
    - Number of filters per layer
    - Use of batch normalization and dropout
    - Number of classes for classification
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        num_filters: list = None,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.5,
        input_size: int = 32
    ):
        """
        Args:
            num_classes: Number of output classes for classification
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            num_filters: List of filter counts for each conv layer (e.g., [32, 64, 128])
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout probability (0 = no dropout, 1 = drop everything)
            input_size: Expected input image size (assumes square images)
        """
        super(SimpleCNN, self).__init__()
        
        # Default filter configuration if none provided
        # Pattern: increase filters as spatial dimensions decrease
        # This maintains roughly constant computational cost per layer
        if num_filters is None:
            num_filters = [32, 64, 128]
        
        self.num_filters = num_filters
        self.use_batchnorm = use_batchnorm
        
        # Build convolutional layers dynamically
        self.features = self._build_feature_extractor(in_channels, num_filters)
        
        # Calculate the size after convolutions for the classifier
        # Each MaxPool2d reduces spatial dimensions by half
        feature_size = input_size // (2 ** len(num_filters))
        flattened_size = num_filters[-1] * feature_size * feature_size
        
        # Classifier head with dropout for regularization
        # Dropout randomly zeros out neurons during training to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),  # Regularization: randomly zero 50% of neurons
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Alternative: Global Average Pooling (GAP)
        # Instead of flattening, take average across spatial dimensions
        # This reduces parameters and provides some spatial invariance
        self.use_gap = False  # Toggle between flatten and GAP
        if self.use_gap:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Output: [B, C, 1, 1]
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_filters[-1], num_classes)
            )
    
    def _build_feature_extractor(self, in_channels: int, num_filters: list) -> nn.Sequential:
        """
        Build the feature extraction layers.
        
        Pattern: Each layer doubles the channels and halves the spatial dimensions
        via max pooling. This is a common pattern in CNNs (VGG, ResNet, etc.)
        """
        layers = []
        current_channels = in_channels
        
        for i, num_filter in enumerate(num_filters):
            # Use pooling on all but the last layer (or adjust as needed)
            use_pool = True
            
            layers.append(ConvBlock(
                in_channels=current_channels,
                out_channels=num_filter,
                kernel_size=3,
                stride=1,
                padding=1,
                use_batchnorm=self.use_batchnorm,
                activation="relu",
                use_pool=use_pool,
                pool_size=2
            ))
            
            current_channels = num_filter
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Flow: Input -> Feature Extractor -> Classifier -> Output
        """
        # Extract features
        x = self.features(x)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection (ResNet-style).
    
    Skip connections allow gradients to flow directly through the network,
    enabling training of very deep networks. This was a breakthrough
    introduced in "Deep Residual Learning for Image Recognition" (2015).
    
    The key idea: instead of learning F(x), learn F(x) + x (the residual)
    """
    
    def __init__(self, channels: int, use_batchnorm: bool = True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with skip connection: output = F(x) + x"""
        identity = x  # Save input for skip connection
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection: add input to output
        out += identity
        out = self.relu(out)
        
        return out


class CNNWithResidual(nn.Module):
    """
    CNN with residual connections for better gradient flow.
    
    This architecture combines the SimpleCNN pattern with residual blocks,
    making it easier to train deeper networks.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        num_filters: list = None,
        num_residual_blocks: int = 2
    ):
        super(CNNWithResidual, self).__init__()
        
        if num_filters is None:
            num_filters = [64, 128]
        
        # Initial convolution
        self.initial_conv = ConvBlock(
            in_channels, num_filters[0],
            kernel_size=7, stride=2, padding=3,
            use_pool=False
        )
        
        # Residual blocks
        self.residual_layers = nn.ModuleList([
            ResidualBlock(num_filters[0]) for _ in range(num_residual_blocks)
        ])
        
        # Additional conv layers
        self.additional_convs = nn.Sequential()
        for i in range(1, len(num_filters)):
            self.additional_convs.add_module(
                f"conv_{i}",
                ConvBlock(num_filters[i-1], num_filters[i], use_pool=True)
            )
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_filters[-1], num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        
        # Apply residual blocks
        for block in self.residual_layers:
            x = block(x)
        
        x = self.additional_convs(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_cnn_model(
    model_type: str = "simple",
    num_classes: int = 10,
    in_channels: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CNN models.
    
    Args:
        model_type: 'simple' or 'residual'
        num_classes: Number of output classes
        in_channels: Number of input channels
        **kwargs: Additional arguments passed to the model constructor
    
    Returns:
        A PyTorch CNN model
    """
    if model_type == "simple":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels, **kwargs)
    elif model_type == "residual":
        return CNNWithResidual(num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # ============================================================
    # Demonstration of the CNN models
    # ============================================================
    
    print("=" * 60)
    print("CNN Example - Common CNN Patterns in PyTorch")
    print("=" * 60)
    
    # Example 1: Simple CNN for CIFAR-10 (32x32 images)
    print("\n1. Simple CNN for CIFAR-10 (32x32 RGB images, 10 classes)")
    print("-" * 60)
    
    cifar_model = SimpleCNN(
        num_classes=10,
        in_channels=3,
        num_filters=[32, 64, 128],
        use_batchnorm=True,
        dropout_rate=0.5,
        input_size=32
    )
    
    print(f"Model architecture:\n{cifar_model}")
    print(f"\nTotal trainable parameters: {cifar_model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 32, 32)
    cifar_model.eval()
    with torch.no_grad():
        output = cifar_model(test_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (first sample): {output[0].numpy()}")
    
    # Example 2: Simple CNN for MNIST (28x28 grayscale)
    print("\n" + "=" * 60)
    print("2. Simple CNN for MNIST (28x28 grayscale, 10 classes)")
    print("-" * 60)
    
    mnist_model = SimpleCNN(
        num_classes=10,
        in_channels=1,
        num_filters=[16, 32],
        use_batchnorm=True,
        dropout_rate=0.3,
        input_size=28
    )
    
    print(f"Total parameters: {mnist_model.count_parameters():,}")
    
    test_input_mnist = torch.randn(2, 1, 28, 28)
    mnist_model.eval()
    with torch.no_grad():
        output_mnist = mnist_model(test_input_mnist)
    print(f"Input: {test_input_mnist.shape} -> Output: {output_mnist.shape}")
    
    # Example 3: CNN with Residual Connections
    print("\n" + "=" * 60)
    print("3. CNN with Residual Connections")
    print("-" * 60)
    
    residual_model = CNNWithResidual(
        num_classes=100,
        in_channels=3,
        num_filters=[64, 128, 256],
        num_residual_blocks=3
    )
    
    print(f"Total parameters: {residual_model.count_parameters():,}")
    
    test_input_res = torch.randn(2, 3, 224, 224)
    residual_model.eval()
    with torch.no_grad():
        output_res = residual_model(test_input_res)
    print(f"Input: {test_input_res.shape} -> Output: {output_res.shape}")
    
    # Example 4: Using the factory function
    print("\n" + "=" * 60)
    print("4. Using the Factory Function")
    print("-" * 60)
    
    model_simple = create_cnn_model(
        model_type="simple",
        num_classes=10,
        num_filters=[32, 64]
    )
    print(f"Simple CNN parameters: {model_simple.count_parameters():,}")
    
    model_residual = create_cnn_model(
        model_type="residual",
        num_classes=10,
        num_filters=[32, 64]
    )
    print(f"Residual CNN parameters: {model_residual.count_parameters():,}")
    
    print("\n" + "=" * 60)
    print("Summary of Key CNN Components Demonstrated:")
    print("=" * 60)
    print("✓ Conv2d: Feature extraction with learnable filters")
    print("✓ MaxPool2d: Spatial downsampling and translation invariance")
    print("✓ BatchNorm2d: Training stabilization and faster convergence")
    print("✓ Dropout: Regularization to prevent overfitting")
    print("✓ ReLU: Non-linear activation function")
    print("✓ Skip Connections: Better gradient flow in deep networks")
    print("✓ Global Average Pooling: Parameter reduction alternative to flattening")
    print("=" * 60)
