#!/usr/bin/env python3
"""Generate AlexNet tutorial images"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

print("Loading model...")
model = AlexNet = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
)

# Initialize weights
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

print("Creating full model...")
full_model = nn.Sequential(
    model,
    nn.AdaptiveAvgPool2d((6, 6)),
    Classifier()
)

total_params = sum(p.numel() for p in full_model.parameters())
print(f"Total parameters: {total_params:,}")

# 1. Layer sizes
print("\n1. Generating alexnet_layers.png...")
layer_sizes = [
    ("Input", 224, 224, 3),
    ("Conv1+Pool", 27, 27, 96),
    ("Conv2+Pool", 13, 13, 256),
    ("Conv3", 13, 13, 384),
    ("Conv4", 13, 13, 384),
    ("Conv5+Pool", 6, 6, 256),
    ("FC1", 1, 1, 4096),
    ("FC2", 1, 1, 4096),
    ("Output", 1, 1, 1000),
]

fig, ax = plt.subplots(figsize=(14, 8))
for i, (name, h, w, c) in enumerate(layer_sizes):
    rect = plt.Rectangle((i*1.5, 0), 1, h/10, facecolor=plt.cm.viridis(i/8), edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(i*1.5 + 0.5, h/10 + 0.5, f"{name}\n{h}x{w}x{c}", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlim(-0.5, len(layer_sizes)*1.5)
ax.set_ylim(-2, 25)
ax.set_xlabel('Network Layers', fontsize=12)
ax.set_title('AlexNet Layer Output Sizes', fontsize=14, fontweight='bold')
ax.set_yticks([])
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig('alexnet_layers.png', dpi=150)
plt.close()
print("  Saved: alexnet_layers.png")

# 2. Conv kernels
print("2. Generating alexnet_filters.png...")
conv1_weights = model[0].weight.data

fig, axes = plt.subplots(8, 12, figsize=(16, 12))
axes = axes.flatten()

for i in range(96):
    weight = conv1_weights[i].cpu().numpy()
    weight = np.transpose(weight, (1, 2, 0))
    weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
    axes[i].imshow(weight)
    axes[i].axis('off')

plt.suptitle('AlexNet Conv1 Kernels (96 filters, 11x11 each)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('alexnet_filters.png', dpi=150)
plt.close()
print("  Saved: alexnet_filters.png")

# 3. Parameters
print("3. Generating alexnet_parameters.png...")
conv_params = sum(p.numel() for name, p in full_model.named_parameters() if 'Conv' in name)
fc_params = sum(p.numel() for name, p in full_model.named_parameters() if 'Linear' in name)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sizes = [conv_params, fc_params]
labels = [f'Conv Layers\n{conv_params/1e6:.1f}M', f'FC Layers\n{fc_params/1e6:.1f}M']
colors = ['#FF6B6B', '#4ECDC4']

ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('AlexNet Parameter Distribution', fontsize=14, fontweight='bold')

# Layer params
layer_params = []
layer_names = []
for name, module in full_model.named_modules():
    if len(list(module.children())) == 0:
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            layer_names.append(name)
            layer_params.append(params)

top_layers = layer_params[:15]
top_names = layer_names[:15]
colors_bar = ['#FF6B6B' if 'Conv' in n else '#4ECDC4' for n in top_names]
ax2.barh(range(len(top_layers)), top_layers, color=colors_bar)
ax2.set_yticks(range(len(top_layers)))
ax2.set_yticklabels([n.split('.')[-1] for n in top_names], fontsize=8)
ax2.set_xlabel('Parameters')
ax2.set_title('Parameters per Layer (Top 15)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('alexnet_parameters.png', dpi=150)
plt.close()
print("  Saved: alexnet_parameters.png")

# 4. Innovations
print("4. Generating alexnet_innovations.png...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ReLU
ax1 = axes[0, 0]
x = np.linspace(-6, 6, 100)
sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)
ax1.plot(x, sigmoid, 'b-', linewidth=2, label='Sigmoid')
ax1.plot(x, relu, 'r-', linewidth=2, label='ReLU')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('1. ReLU Activation\n(Faster, Sparse)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Dropout
ax2 = axes[0, 1]
n_neurons = 20
prob_active = np.random.random(n_neurons) > 0.5
colors_d = ['green' if p else 'red' for p in prob_active]
ax2.barh(range(n_neurons), [1]*n_neurons, color=colors_d, height=0.8)
ax2.set_xlim(0, 1.5)
ax2.set_yticks([])
ax2.set_xlabel('Neuron State')
ax2.set_title('2. Dropout Regularization\n(Random 50% OFF)', fontsize=12, fontweight='bold')
ax2.text(1.2, 10, 'Green: Active\nRed: Dropped', fontsize=10)

# Data Aug
ax3 = axes[1, 0]
img = np.random.rand(100, 100, 3)
ax3.imshow(img)
ax3.add_patch(plt.Rectangle((10, 10), 80, 80, fill=False, edgecolor='red', linewidth=2))
ax3.set_title('3. Data Augmentation\n(Random Crop, Flip)', fontsize=12, fontweight='bold')
ax3.axis('off')

# GPU
ax4 = axes[1, 1]
ax4.bar(['CPU', 'GPU'], [100, 8], color=['#FF6B6B', '#4ECDC4'])
ax4.set_ylabel('Relative Training Time')
ax4.set_title('4. GPU Parallel\n(~12x Speedup)', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 120)
for i, v in enumerate([100, 8]):
    ax4.text(i, v + 3, f'{v}x', ha='center', fontsize=12, fontweight='bold')

plt.suptitle('AlexNet Four Key Innovations', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('alexnet_innovations.png', dpi=150)
plt.close()
print("  Saved: alexnet_innovations.png")

# 5. ImageNet History
print("5. Generating alexnet_imagenet_history.png...")
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
error_rates = [28.2, 25.8, 15.3, 14.8, 10.0, 6.7, 5.9, 5.3]
methods = ['SIFT+FVs', 'CNNs', 'AlexNet', 'ZFNet', 'VGG/GoogleNet', 'ResNet', 'ResNet', 'SENet']

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['gray', 'red', '#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4', '#45B7D1', '#45B7D1']
bars = ax.bar(years, error_rates, color=colors, edgecolor='black', linewidth=1.5)

for bar, method, rate in zip(bars, methods, error_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{rate}%\n{method}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Top-5 Error Rate (%)', fontsize=12)
ax.set_title('ImageNet Top-5 Error Rate History\n(AlexNet Started Deep Learning Era)', fontsize=14, fontweight='bold')
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Human Level (~5%)')
ax.legend()
ax.set_ylim(0, 35)
ax.annotate('AlexNet!', xy=(2012, 15.3), xytext=(2012.5, 22),
            fontsize=12, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('alexnet_imagenet_history.png', dpi=150)
plt.close()
print("  Saved: alexnet_imagenet_history.png")

# 6. Transfer Learning
print("6. Generating alexnet_transfer.png...")
trainable = 4096 * 10 + 10
frozen = 62300000 - trainable

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(['Trainable', 'Frozen'], [trainable/1e6, frozen/1e6], color=['#4ECDC4', '#FF6B6B'])
ax.set_ylabel('Parameters (Millions)')
ax.set_title('Transfer Learning: Fine-tune Only Small Part', fontsize=14, fontweight='bold')
for i, v in enumerate([trainable/1e6, frozen/1e6]):
    ax.text(i, v + 1, f'{v:.2f}M', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('alexnet_transfer.png', dpi=150)
plt.close()
print("  Saved: alexnet_transfer.png")

print("\n✅ All images generated!")
print("\nFiles created:")
import os
for f in sorted(os.listdir('.')):
    if f.endswith('.png'):
        print(f"  📷 {f}")