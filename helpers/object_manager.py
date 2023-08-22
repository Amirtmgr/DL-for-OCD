import os
import directory_manager as dm
import pickle as pk

# Function to save object to file
def save_object(obj, folder, file):
    
    folder_path = dm.create_folder(folder)
    
    file_path = os.path.join(folder_path, file.rsplit( ".", 1 )[0] + ".pt")
    
    try:
        with open(file_path, 'wb') as file:
            pk.dump(obj, file)
        return "Object has been successfully saved to file: " + file_path
    except Exception as e:
        return "Error encountered while saving the object: " + str(e)

# Function to load object
def load_object(folder, file):
    
    folder_path = dm.create_folder(folder)
    file_path = os.path.join(folder_path, file.rsplit( ".", 1 )[0] + ".pt")
    
    try:
        with open(file_path, 'rb') as file:
            obj = pk.load(file)
        return obj
    except Exception as e:
        return "Error encountered while loading the object: " + str(e)
