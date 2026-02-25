import unittest
import sys
import os

# Ensure the root project directory is in the Python path so we can import 'modules'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.search_space import LayersBased

class TestLayersBasedSearchSpace(unittest.TestCase):
    def setUp(self):
        """Set up a dummy configuration to initialize the search space before each test."""
        self.config = {
            "layers_types": ["Conv2d", "MaxPool2d", "Dropout", "ReLU"],
            "layers_count": [5, 7],
            "channels": [3, 5],
            "kernel": [2, 3],
            "padding": [0, 1, 2],
            "last_hid_mlp": [0, 50, 100],
            "dropout_rates": [0.1, 0.3]
        }
        self.search_space = LayersBased()
        self.search_space.define_space(self.config)

    def test_no_consecutive_dropouts(self):
        """
        Crucial Test: Ensures that 'Dropout' is never followed immediately 
        by another 'Dropout' layer.
        """
        # Run 100 times to ensure random generation doesn't slip up
        for _ in range(100):
            arch = self.search_space.sample_architecture()
            layers = arch["layers"]
            
            for i in range(len(layers) - 1):
                current_layer_type = layers[i]["type"]
                next_layer_type = layers[i+1]["type"]
                
                # Assert that it is FALSE that both current and next are Dropout
                self.assertFalse(
                    current_layer_type == "Dropout" and next_layer_type == "Dropout",
                    f"Invalid architecture generated! Consecutive Dropouts found: {layers}"
                )

    def test_valid_layer_count(self):
        """Test that the number of generated layers respects the config."""
        for _ in range(50):
            arch = self.search_space.sample_architecture()
            self.assertIn(
                len(arch["layers"]), 
                self.config["layers_count"],
                "Generated architecture has an invalid number of layers."
            )

    def test_layer_parameters(self):
        """Test that layer-specific parameters are properly sampled from the config space."""
        arch = self.search_space.sample_architecture()
        
        for layer in arch["layers"]:
            l_type = layer["type"]
            self.assertIn(l_type, self.config["layers_types"])
            
            if l_type == "Conv2d":
                self.assertIn(layer["channels"], self.config["channels"])
                self.assertIn(layer["kernel"], self.config["kernel"])
                self.assertIn(layer["padding"], self.config["padding"])
            elif l_type == "Dropout":
                self.assertIn(layer["rate"], self.config["dropout_rates"])

    def test_last_hid_mlp(self):
        """Test that the final MLP hidden layer parameter is valid."""
        arch = self.search_space.sample_architecture()
        self.assertIn(arch["last_hid_mlp"], self.config["last_hid_mlp"])

if __name__ == '__main__':
    # Determine the path to save the text file (inside the tests directory)
    report_path = os.path.join(os.path.dirname(__file__), 'test_report.txt')
    
    # Open the text file and write the test results into it
    with open(report_path, 'w') as f:
        f.write("=========================================\n")
        f.write("Unit Test Execution Report\n")
        f.write("Project: CNN_Cls_miniNAS\n")
        f.write("=========================================\n\n")
        
        # Use TextTestRunner to redirect the test output stream to our file
        # verbosity=2 gives us the detailed "test_name ... OK" format
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        
        # exit=False prevents unittest from closing the whole python script
        # before our final print statement runs.
        unittest.main(testRunner=runner, exit=False)
        
    print(f"Test execution complete. Detailed report saved to: {report_path}")