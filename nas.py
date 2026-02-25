import argparse
from utils import io_ops
from modules.search_space import LayersBased
from modules.validation_strategy import FullTraining
from modules.search_strategy import RandomSearch

def generate_pytorch_code(best_arch: dict, dataset_name="MNIST") -> str:
    """
    Translates the winning architecture dictionary into executable PyTorch code.
    Hardcoded defaults are set for the MNIST dataset.
    """
    # MNIST Specifics
    num_classes = 10
    in_channels = 1
    input_h, input_w = 28, 28

    lines = [
        f"# Automatically generated CNN Architecture for {dataset_name}",
        "import torch",
        "import torch.nn as nn",
        "",
        "class WinningModel(nn.Module):",
        f"    def __init__(self, num_classes={num_classes}):",
        "        super(WinningModel, self).__init__()",
        "        self.features = nn.Sequential("
    ]
    
    current_channels = in_channels
    layers = best_arch.get("layers", [])
    
    # 1. Build the feature extractor
    for layer in layers:
        l_type = layer["type"]
        if l_type == "Conv2d":
            out_c = layer["channels"]
            k = layer["kernel"]
            p = layer["padding"]
            lines.append(f"            nn.Conv2d(in_channels={current_channels}, out_channels={out_c}, kernel_size={k}, padding={p}),")
            current_channels = out_c
            
        elif l_type == "MaxPool2d":
            lines.append(f"            nn.MaxPool2d(kernel_size=2, stride=2),")
            
        elif l_type == "ReLU":
            lines.append(f"            nn.ReLU(),")
            
        elif l_type == "Dropout":
            r = layer.get("rate", 0.5)
            lines.append(f"            nn.Dropout(p={r}),")
            
    lines.append("        )")
    lines.append("")
    
    # 2. Build the classifier head
    lines.append("        self.classifier = nn.Sequential(")
    lines.append("            nn.Flatten(),")
    
    last_hid = best_arch.get("last_hid_mlp", 0)
    if last_hid > 0:
        lines.append(f"            nn.LazyLinear(out_features={last_hid}),")
        lines.append("            nn.ReLU(),")
        lines.append(f"            nn.Linear(in_features={last_hid}, out_features=num_classes)")
    else:
        lines.append("            nn.LazyLinear(out_features=num_classes)")
        
    lines.append("        )")
    lines.append("")
    
    # 3. Add the forward pass
    lines.append("    def forward(self, x):")
    lines.append("        x = self.features(x)")
    lines.append("        x = self.classifier(x)")
    lines.append("        return x")
    lines.append("")
    
    # 4. Add a test block specific to MNIST dimensions
    lines.append(f"# --- Test the generated model with {dataset_name} dimensions ---")
    lines.append("if __name__ == '__main__':")
    lines.append("    model = WinningModel()")
    lines.append(f"    # Dummy forward pass (Batch Size 1, {in_channels} Channel, {input_h}x{input_w} Image)")
    lines.append(f"    dummy_input = torch.randn(1, {in_channels}, {input_h}, {input_w})")
    lines.append("    output = model(dummy_input) # Initializes LazyLinear")
    lines.append("    print(model)")
    lines.append("    print(f'Output shape: {output.shape} (Expected: [1, {num_classes}])')")
    
    return "\n".join(lines)


def architecture_search(config_path: str):
    """Main NAS orchestration function."""
    print("1. Loading Config...")
    config = io_ops.load_config(config_path)
    
    print("2. Defining Search Space...")
    search_space_impl = LayersBased()
    space_dict = search_space_impl.define_space(config['SearchSpace'])
    
    print("3. Initializing Validation Strategy...")
    # The FullTraining validation strategy will handle the actual MNIST dataset loading 
    # and training loop using the parameters specified in the config.
    val_strategy = FullTraining(config['ValidationStrategy'])
    
    print("4. Executing Architecture Search...")
    iterations = config['SearchStrategy']['nbr_iterations']
    search_strategy = RandomSearch(iterations, val_strategy, search_space_impl)
    results = search_strategy.generate_architecture(space_dict)
    
    print("\n5. Generating PyTorch Code for MNIST & Saving Outputs...")
    best_arch = results['best_architecture']
    
    # Generate the PyTorch code using the MNIST-specific generator
    code_str = generate_pytorch_code(best_arch, dataset_name="MNIST")
    
    model_and_stats = {
        "best_metrics": results['best_metrics'],
        "code": code_str,
        "history": results['history']
    }
    
    # Save the artifacts (images, text file, and the python code file)
    io_ops.save_model("./outputs", model_and_stats)
    print("Search Complete. Please check the './outputs' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN_Cls_miniNAS Execution")
    parser.add_argument("config", help="Path to the config file (e.g., ./configs/config1.yaml)")
    args = parser.parse_args()
    
    architecture_search(args.config)