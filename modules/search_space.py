from abc import ABC, abstractmethod
import random

class SearchSpace(ABC):
    @abstractmethod
    def define_space(self, parameters: dict) -> dict:
        pass

class LayersBased(SearchSpace):
    def __init__(self):
        self.space = {}

    def define_space(self, parameters: dict) -> dict:
        self.space = parameters
        return self.space
        
    def sample_architecture(self) -> dict:
        """Helper to generate a random valid architecture from the defined space."""
        layers_count = random.choice(self.space['layers_count'])
        arch_layers = []
        last_was_dropout = False
        
        for _ in range(layers_count):
            valid_types = list(self.space['layers_types'])
            # Rule: do not generate consecutive dropouts
            if last_was_dropout and 'Dropout' in valid_types:
                valid_types.remove('Dropout')
                
            layer_type = random.choice(valid_types)
            layer_def = {'type': layer_type}
            
            if layer_type == 'Conv2d':
                layer_def['channels'] = random.choice(self.space['channels'])
                layer_def['kernel'] = random.choice(self.space['kernel'])
                layer_def['padding'] = random.choice(self.space['padding'])
            elif layer_type == 'Dropout':
                layer_def['rate'] = random.choice(self.space['dropout_rates'])
                
            arch_layers.append(layer_def)
            last_was_dropout = (layer_type == 'Dropout')
            
        return {
            "layers": arch_layers, 
            "last_hid_mlp": random.choice(self.space['last_hid_mlp'])
        }