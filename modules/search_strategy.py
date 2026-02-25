from abc import ABC, abstractmethod
from modules.search_space import LayersBased
from modules.validation_strategy import ValidationStrategy

class SearchStrategy(ABC):
    @abstractmethod
    def generate_architecture(self, search_space: dict) -> dict:
        pass

class RandomSearch(SearchStrategy):
    def __init__(self, nbr_iterations: int, validator: ValidationStrategy, space_generator: LayersBased):
        self.nbr_iterations = nbr_iterations
        self.validator = validator
        self.space_generator = space_generator
        
    def generate_architecture(self, search_space: dict) -> dict:
        best_arch = None
        best_metrics = {"acc_val": -1}
        history = []
        
        for i in range(self.nbr_iterations):
            print(f"\n--- Model {i+1}/{self.nbr_iterations} ---")
            
            # Generate and validate
            arch = self.space_generator.sample_architecture()
            metrics = self.validator.validate(arch)
            
            history.append({"architecture": arch, "metrics": metrics})
            
            # Track best model
            if metrics["acc_val"] > best_metrics["acc_val"]:
                best_metrics = metrics
                best_arch = arch
                
        return {
            "best_architecture": best_arch, 
            "best_metrics": best_metrics, 
            "history": history
        }