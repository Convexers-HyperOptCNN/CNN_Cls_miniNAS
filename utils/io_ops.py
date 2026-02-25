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