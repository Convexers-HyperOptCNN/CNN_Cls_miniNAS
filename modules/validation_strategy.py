from abc import ABC, abstractmethod
import random

class ValidationStrategy(ABC):
    @abstractmethod
    def validate(self, architecture: dict) -> dict:
        pass

class FullTraining(ValidationStrategy):
    def __init__(self, config: dict):
        self.config = config
        
    def validate(self, architecture: dict) -> dict:
        """Trains the model and returns metrics."""
        print(f"Training architecture with {len(architecture['layers'])} layers...")
        
        # TODO: Implement actual PyTorch/TensorFlow data loading and training loop here using MNIST
        
        # MOCK DATA for testing the pipeline
        epochs = self.config['epochs']
        mock_acc = random.uniform(0.60, 0.98)
        mock_loss_history = [random.uniform(0.1, 2.0) / (e + 1) for e in range(epochs)]
        mock_runtime = random.uniform(10.0, 60.0)
        
        return {
            "acc_val": mock_acc,
            "runtime": mock_runtime,
            "loss_history": mock_loss_history
        }