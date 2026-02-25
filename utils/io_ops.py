import yaml
import os
import matplotlib.pyplot as plt

def load_config(path: str) -> dict:
    """Loads and parses the configuration file."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def save_model(path: str, model_and_stats: dict):
    """Saves loss history images, accuracies, and the winning architecture code."""
    os.makedirs(path, exist_ok=True)
    
    # 1. Save winning architecture code
    code_path = os.path.join(path, "winning_model.py")
    with open(code_path, "w") as f:
        f.write(model_and_stats["code"])
        
    # 2. Save accuracies to .txt
    acc_path = os.path.join(path, "accuracies.txt")
    with open(acc_path, "w") as f:
        f.write(f"Best Accuracy: {model_and_stats['best_metrics']['acc_val']:.4f}\n\n")
        f.write("All Iterations:\n")
        for idx, stat in enumerate(model_and_stats["history"]):
            f.write(f"Model {idx+1}: {stat['metrics']['acc_val']:.4f}\n")
            
    # 3. Save loss history as images
    loss_dir = os.path.join(path, "loss_histories")
    os.makedirs(loss_dir, exist_ok=True)
    
    for idx, stat in enumerate(model_and_stats["history"]):
        plt.figure()
        plt.plot(stat['metrics']['loss_history'], label='Train Loss')
        plt.title(f"Model {idx+1} Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(loss_dir, f"model_{idx+1}_loss.png"))
        plt.close()

    print(f"Artifacts successfully saved to {path}/")

def generate_pytorch_code(best_arch: dict) -> str:
    """
    Translates a dictionary defining a CNN architecture into a string 
    of valid, executable PyTorch code.
    """
    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "class WinningModel(nn.Module):",
        "    def __init__(self, num_classes=10):",
        "        super(WinningModel, self).__init__()",
        "        self.features = nn.Sequential("
    ]
    
    in_channels = 1 # MNIST has 1 color channel (grayscale)
    layers = best_arch.get("layers", [])
    
    # 1. Build the feature extractor (CNN layers)
    for i, layer in enumerate(layers):
        l_type = layer["type"]
        if l_type == "Conv2d":
            out_c = layer["channels"]
            k = layer["kernel"]
            p = layer["padding"]
            lines.append(f"            nn.Conv2d(in_channels={in_channels}, out_channels={out_c}, kernel_size={k}, padding={p}),")
            in_channels = out_c # Update input channels for the next layer
            
        elif l_type == "MaxPool2d":
            # Default pool size and stride of 2
            lines.append(f"            nn.MaxPool2d(kernel_size=2, stride=2),")
            
        elif l_type == "ReLU":
            lines.append(f"            nn.ReLU(),")
            
        elif l_type == "Dropout":
            r = layer.get("rate", 0.5)
            lines.append(f"            nn.Dropout(p={r}),")
            
    lines.append("        )")
    lines.append("")
    
    # 2. Build the classifier head (MLP layers)
    lines.append("        self.classifier = nn.Sequential(")
    lines.append("            nn.Flatten(),")
    
    # Use LazyLinear to avoid manually calculating the flattened spatial dimensions
    last_hid = best_arch.get("last_hid_mlp", 0)
    if last_hid > 0:
        lines.append(f"            nn.LazyLinear(out_features={last_hid}),")
        lines.append("            nn.ReLU(),")
        lines.append(f"            nn.Linear(in_features={last_hid}, out_features=num_classes)")
    else:
        # Direct mapping to classes if no hidden MLP layer
        lines.append("            nn.LazyLinear(out_features=num_classes)")
        
    lines.append("        )")
    lines.append("")
    
    # 3. Add the forward pass method
    lines.append("    def forward(self, x):")
    lines.append("        x = self.features(x)")
    lines.append("        x = self.classifier(x)")
    lines.append("        return x")
    lines.append("")
    lines.append("# --- Test the generated model ---")
    lines.append("if __name__ == '__main__':")
    lines.append("    model = WinningModel()")
    lines.append("    # Dummy forward pass (Batch Size 1, 1 Channel, 28x28 Image for MNIST)")
    lines.append("    dummy_input = torch.randn(1, 1, 28, 28)")
    lines.append("    output = model(dummy_input) # Initializes LazyLinear")
    lines.append("    print(model)")
    lines.append("    print(f'Output shape: {output.shape}')")
    
    return "\n".join(lines)