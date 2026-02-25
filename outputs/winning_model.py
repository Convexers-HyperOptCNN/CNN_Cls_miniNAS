# Automatically generated CNN Architecture for MNIST
import torch
import torch.nn as nn

class WinningModel(nn.Module):
    def __init__(self, num_classes=10):
        super(WinningModel, self).__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, padding=2),
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Test the generated model with MNIST dimensions ---
if __name__ == '__main__':
    model = WinningModel()
    # Dummy forward pass (Batch Size 1, 1 Channel, 28x28 Image)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input) # Initializes LazyLinear
    print(model)
    print(f'Output shape: {output.shape} (Expected: [1, {num_classes}])')