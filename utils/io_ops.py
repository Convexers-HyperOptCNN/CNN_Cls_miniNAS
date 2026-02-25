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
        
    # Sort history by accuracy (highest to lowest) to find the best combinations
    sorted_history = sorted(
        model_and_stats["history"], 
        key=lambda x: x["metrics"]["acc_val"], 
        reverse=True
    )
    
    # --- PRINT TO TERMINAL ---
    print("\n=========================================")
    print("🏆 TOP 3 BEST ARCHITECTURE COMBINATIONS 🏆")
    print("=========================================")
    for rank, stat in enumerate(sorted_history[:3]): # Print top 3
        acc = stat['metrics']['acc_val']
        layers = [layer['type'] for layer in stat['architecture']['layers']]
        print(f"Rank {rank+1}: Accuracy = {acc:.4f}")
        print(f"Layers: {' -> '.join(layers)}")
        print(f"MLP Hidden: {stat['architecture']['last_hid_mlp']}\n")
        
    # 2. Save detailed combinations to accuracies.txt
    acc_path = os.path.join(path, "accuracies.txt")
    with open(acc_path, "w") as f:
        f.write("=========================================\n")
        f.write("NAS Search Results - Best Combinations\n")
        f.write("=========================================\n\n")
        
        f.write(f"Best Overall Accuracy: {model_and_stats['best_metrics']['acc_val']:.4f}\n\n")
        
        f.write("All Iterations (Ranked Best to Worst):\n")
        f.write("-" * 40 + "\n")
        
        for rank, stat in enumerate(sorted_history):
            f.write(f"Rank {rank+1} | Accuracy: {stat['metrics']['acc_val']:.4f} | Runtime: {stat['metrics']['runtime']:.2f}s\n")
            f.write("Combination:\n")
            
            # Write out the detailed parameters of each layer
            for i, layer in enumerate(stat["architecture"]["layers"]):
                layer_details = ", ".join([f"{k}={v}" for k, v in layer.items() if k != "type"])
                f.write(f"  Layer {i+1}: {layer['type']} ({layer_details})\n")
                
            f.write(f"  Final MLP Hidden Layer: {stat['architecture']['last_hid_mlp']}\n")
            f.write("-" * 40 + "\n")
            
    # 3. Save loss history as images
    loss_dir = os.path.join(path, "loss_histories")
    os.makedirs(loss_dir, exist_ok=True)
    
    # We use the original history list to keep the "Model 1, Model 2" chronological naming for images
    for idx, stat in enumerate(model_and_stats["history"]):
        plt.figure()
        plt.plot(stat['metrics']['loss_history'], label='Train Loss')
        plt.title(f"Model {idx+1} Loss History\nAcc: {stat['metrics']['acc_val']:.4f}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(loss_dir, f"model_{idx+1}_loss.png"))
        plt.close()

    print(f"\nArtifacts successfully saved to {path}/")