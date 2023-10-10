# ------------------------------------------------------------------------
# Description: Utilies Module
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import os
import yaml
import easydict
import datetime as dt

# TODO(atm):Review Class

class __ConfigLoader:
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
        self.file_name = None
        self.config_dict = {}
        self.config = {}
        self.main_path = None
    
    def load_config(self, file_name):
        """Method to load config file
        Args:
            file_name (str): Config file
        
        Raises:
            FileNotFoundError: If file not found.
        
        """
        try:
            with open(file_name, 'r') as file:
                self.file_name = file_name.split("/")[-1].split(".")[0]
                self.config_dict = yaml.safe_load(file)
                temp = self._set_paths(self.config_dict)
                temp['file_name'] = self.file_name
                self.config = easydict.EasyDict(temp)    
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{file_name}' not found.")

    def set_main_path(self, main_path):
        """Method to set main path
        Args:
            main_path (str): Main path
        """
        self.main_path = main_path

    def _set_paths(self, temp):
        """Method to set other paths
        """
        temp["main_path"] = self.main_path
        folder = dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        print("Folder: ", folder)
        temp["folder"] = folder
        temp["models_folder"] = os.path.join(self.main_path,"saved", "models")
        temp["results_folder"] = os.path.join(self.main_path,"saved", "results")
        temp["logs_folder"] = os.path.join(self.main_path,"saved", "logs")
        temp["charts_folder"] = os.path.join(self.main_path,"saved", "charts")
        
        temp["models_path"] = os.path.join(self.main_path,"saved", "models",folder)
        temp["results_path"] = os.path.join(self.main_path,"saved", "results", folder)
        temp["logs_path"] = os.path.join(self.main_path,"saved", "logs", folder)
        temp["charts_path"] = os.path.join(self.main_path,"saved", "charts", folder)
        temp["data_path"] = os.path.join(self.main_path,"data")
        #TODO: Add other paths like dataset paths,etc.
        return temp

    def get_variable(self, variable_name):
        """Method to get values of the specific variable
        Args:
            variable_name (str): Variable name whose value is required.
        
        Returns:
            Value if found otherwise None Type.
        """
        return self.config.get(variable_name)

    def export_to_environment(self, keys=None):
        """Export variables as Environment variables
        """
        # TODO(atm):Does not work for nested dict
        
        for key, value in self.config.items():
            if key in keys or keys is None:
                os.environ[key] = str(value)
    
    def print_config_dict(self):
        """Print config dictionary
        """
        print("---------------------" * 5)
        for k, v in self.config_dict.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for k1, v1 in v.items():
                    print(f"\t{k1}: {v1}")
            else:
                print(f"{k}: {v}")

        print("---------------------" * 5)
# Create a singleton instance of ConfigLoader
config_loader = __ConfigLoader()