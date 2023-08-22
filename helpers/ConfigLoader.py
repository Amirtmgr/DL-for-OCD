# ------------------------------------------------------------------------
# Description: Utilies Module
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import os
import yaml

# TODO(atm):Review Class

class ConfigLoader:
    """ConfigLoader class to load config variables.
    
    Typical usage example:
    
        file_name = 'config.yaml'
        try:
            config_loader = ConfigLoader()
            config_loader.load_config(file_name)
            config_loader.export_to_environment()
        except FileNotFoundError as e:
            print(str(e))
    """
    
    def __init__(self):
        self.config = {}
    
    def load_config(self, file_name):
        """Method to load config file
        Args:
            file_name (str): Config file
        
        Raises:
            FileNotFoundError: If file not found.
        
        """
        try:
            with open(file_name, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{file_name}' not found.")

    def get_variable(self, variable_name):
        """Method to get values of the specific variable
        Args:
            variable_name (str): Variable name whose value is required.
        
        Returns:
            Value if found otherwise None Type.
        """
        return self.config.get(variable_name)

    def export_to_environment(self):
        """Export variables as Environment variables
        """
        # TODO(atm):Does not work for nested dict
        
        for variable_name, value in self.config.items():
            os.environ[variable_name] = str(value)
          