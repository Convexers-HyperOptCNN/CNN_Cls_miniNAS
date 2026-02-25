from abc import ABC, abstractmethod
import copy
import random
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
    
class EvolutionarySearch(SearchStrategy):
    def __init__(self, population_size: int, generations: int, validator: ValidationStrategy, space_generator: LayersBased):
        self.population_size = population_size
        self.generations = generations
        self.validator = validator
        self.space_generator = space_generator
        
    def mutate(self, parent_arch: dict) -> dict:
        """Creates a slightly modified copy of the parent architecture."""
        child = copy.deepcopy(parent_arch)
        
        # Randomly choose what to mutate: a layer's hyperparameters OR the final MLP
        if random.random() < 0.3:
            # Mutate the final MLP size
            child['last_hid_mlp'] = random.choice(self.space_generator.space['last_hid_mlp'])
        else:
            # Mutate a random layer's parameter
            layer_to_mutate = random.choice(child['layers'])
            
            if layer_to_mutate['type'] == 'Conv2d':
                # Pick a random property to change
                prop = random.choice(['channels', 'kernel', 'padding'])
                layer_to_mutate[prop] = random.choice(self.space_generator.space[prop])
                
            elif layer_to_mutate['type'] == 'Dropout':
                layer_to_mutate['rate'] = random.choice(self.space_generator.space['dropout_rates'])
                
        return child

    def generate_architecture(self, search_space: dict) -> dict:
        population = []
        history = []
        best_metrics = {"acc_val": -1}
        best_arch = None
        
        print(f"\n--- [Evolutionary Search] Initializing Population ({self.population_size} models) ---")
        # 1. Initialize random population
        for i in range(self.population_size):
            arch = self.space_generator.sample_architecture()
            metrics = self.validator.validate(arch)
            
            model_record = {"architecture": arch, "metrics": metrics}
            population.append(model_record)
            history.append(model_record)
            
        # 2. Evolution Loop
        for gen in range(self.generations):
            print(f"\n--- [Evolutionary Search] Generation {gen + 1}/{self.generations} ---")
            
            # Sort population by accuracy (descending)
            population.sort(key=lambda x: x["metrics"]["acc_val"], reverse=True)
            
            # Select the top 50% as parents
            num_parents = max(1, self.population_size // 2)
            parents = population[:num_parents]
            
            children = []
            # Generate new children to replace the bottom 50%
            for i in range(self.population_size - num_parents):
                parent = random.choice(parents)["architecture"]
                child_arch = self.mutate(parent)
                
                print(f"Evaluating Child {i+1} mutated from parent...")
                child_metrics = self.validator.validate(child_arch)
                
                child_record = {"architecture": child_arch, "metrics": child_metrics}
                children.append(child_record)
                history.append(child_record)
                
            # Create the new population (Top parents + new children)
            population = parents + children
            
        # 3. Find the absolute best from history
        for record in history:
            if record["metrics"]["acc_val"] > best_metrics["acc_val"]:
                best_metrics = record["metrics"]
                best_arch = record["architecture"]
                
        return {
            "best_architecture": best_arch, 
            "best_metrics": best_metrics, 
            "history": history
        }