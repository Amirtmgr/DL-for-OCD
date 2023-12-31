# ------------------------------------------------------------------------
# Description: Utilies Module
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import os
import yaml
import easydict
import datetime as dt
import uuid


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
    
    def load_config(self, file_name, task_type=None, architecture=None):
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
                if task_type:
                    temp['train']['task_type'] = task_type
                
                # Maintain personalization bool with given task type
                temp['dataset']['personalization'] = temp['train']['task_type'] == 5
                
                if architecture:
                    temp['architecture']['name'] = architecture
                
                temp = self._set_paths(self.config_dict)
                
                ftm = 2 if temp['train']['task_type'] == 5 else temp['train']['task_type']
                temp["best_model_folder"] = os.path.join(self.main_path,"saved","best_model", str(ftm),temp['architecture']['name'], temp['optim']['name'])
                temp["best_model_path"] = os.path.join(temp['best_model_folder'], "best_model.pt")
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
        folder = dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + f"-tasktype_{temp['train']['task_type']}-{temp['architecture']['name']}-" +  str(uuid.uuid4().hex)
        print("Folder: ", folder)
        temp["folder"] = folder
        temp["saved_folder"] = os.path.join(self.main_path,"saved", folder)
        temp["models_folder"] = os.path.join(self.main_path,"saved", folder, "models")
        temp["results_folder"] = os.path.join(self.main_path, "results") # To save csv files
        temp["charts_folder"] = os.path.join(self.main_path,"saved", folder, "charts")
        temp["models_path"] = os.path.join(self.main_path,"saved", folder, "models")
        temp["results_path"] = os.path.join(self.main_path,"saved", folder, "results")
        temp["logs_path"] = os.path.join(self.main_path,"saved", folder, "logs") 
        temp["charts_path"] = os.path.join(self.main_path,"saved", folder, "charts")
        temp["data_path"] = os.path.join(self.main_path,"data")
        temp["best_val_loss"] = 0.0
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
        print("Config Dictionary:")
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