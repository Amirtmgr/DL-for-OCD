import os
import datetime
import gc
import numpy as np
import random
from collections import Counter

import torch
from torch.utils.data import ConcatDataset, TensorDataset

from src.dl_pipeline import train as t
from src.helper import data_preprocessing as dp
from src.helper import df_manager as dfm
from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
from src.helper.filters import band_pass_filter
from src.helper import object_manager as om
from src.helper import directory_manager as dm
from src.helper.state import State
from src.helper import data_structures as ds
from src.helper import plotter as pl
from src.helper.metrics import Metrics

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection, NearMiss, RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.pipeline import Pipeline


def run(device, multi_gpu=False):
    print("Device:", device)

    print("======"*20)
    # start
    start = datetime.datetime.now()
    Logger.info(f"Personalization Start time: {start}")
    print(f"Personalization Start time: {start}")

    is_binary = cl.config.dataset.num_classes < 3
    shelf_name = cl.config.dataset.name
    random_seed = cl.config.dataset.random_seed
    shuffle = cl.config.dataset.shuffle
    cv_name = cl.config.train.cross_validation.name
    personalized_subject = str(cl.config.dataset.personalized_subject)
    train_ratio = cl.config.dataset.train_ratio
    test_ratio = cl.config.dataset.test_ratio
    inference_ratio = cl.config.dataset.inference_ratio
    remaining_ratio = 1 - (train_ratio + test_ratio + inference_ratio)
    n_jobs = -1

    if inference_ratio + test_ratio + train_ratio > 1:
        raise ValueError("Total ratio is greater than 1. Adjust the ratios in config file.")


    binary_threshold = cl.config.train.binary_threshold

    # Load python dataset
    X_dict, y_dict = dp.load_shelves(shelf_name, personalized_subject)

    X_personalized = X_dict[personalized_subject]
    y_personalized = y_dict[personalized_subject]

    Logger.info(f"Total samples : | X_personalized shape: {X_personalized.shape} | y_personalized shape: {y_personalized.shape}")

    del X_dict, y_dict
    gc.collect()

    
    #X_infer, X_temp, y_infer, y_temp = train_test_split(X_personalized, y_personalized, train_size= inference_ratio, shuffle=False)

    # Split data
    if shuffle:
        X_infer, X_temp, y_infer, y_temp = train_test_split(X_personalized, y_personalized, train_size= inference_ratio,stratify=y_personalized, shuffle=True, random_state=random_seed)
        X_train, X_temp, y_train, y_temp = train_test_split(X_temp, y_temp, train_size= train_ratio/(train_ratio + test_ratio), stratify = y_temp, shuffle=True, random_state = random_seed)
        if remaining_ratio > 0:
            X_val, X_temp, y_val, y_temp = train_test_split(X_temp, y_temp, train_size= test_ratio/(remaining_ratio + test_ratio), stratify = y_temp, shuffle=True, random_state = random_seed)
        else:
            X_val, y_val = X_temp, y_temp
    else:
        X_infer, X_temp, y_infer, y_temp = train_test_split(X_personalized, y_personalized, train_size= inference_ratio, shuffle=False)
        X_train, X_temp, y_train, y_temp = train_test_split(X_temp, y_temp, train_size= train_ratio/(train_ratio + test_ratio), shuffle=False)
        if remaining_ratio > 0:
            X_val, X_temp, y_val, y_temp = train_test_split(X_temp, y_temp, train_size= test_ratio/(remaining_ratio + test_ratio), shuffle=False)
        else:
            X_val, y_val = X_temp, y_temp

    
    Logger.info(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    Logger.info(f"X_val shape: {X_val.shape} | y_val shape: {y_val.shape}")
    Logger.info(f"X_infer shape: {X_infer.shape} | y_infer shape: {y_infer.shape}")
    Logger.info(f"X_temp shape: {X_temp.shape} | y_temp shape: {y_temp.shape}")

    del X_temp, y_temp
    gc.collect()

   # Sampling
    if cl.config.dataset.sampling:
        samples, window_size, num_features = X_train.shape
        Logger.info(f"===> Before Undersampling {Counter(y_train)}")
        X_reshape = X_train.reshape(samples, -1)       
        # counts = Counter(y_train.astype(int))
        # max_key, max_count = max(counts.items(), key=lambda x:x[1])
        # min_key, min_count = min(counts.items(), key=lambda x:x[1])
        
        # strategy = "majority"
        # resampling_pipeline = Pipeline([
        # ('undersampler', NearMiss(version=3,sampling_strategy=0.5,n_jobs=n_jobs)),
        # ('oversampler', SMOTE(sampling_strategy='auto', random_state=random_seed, n_jobs=n_jobs))
        # ])

        # resampling_pipeline = Pipeline([
        # ('oversampler', SMOTE(sampling_strategy='auto', random_state=random_seed)),
        # ('undersampler', RandomUnderSampler(sampling_strategy='auto', random_state=random_seed))
        # ])

        resampling_pipeline = SMOTE(sampling_strategy='minority', random_state=random_seed, n_jobs=n_jobs)
        
        #resampling_pipeline = NearMiss(version=3,sampling_strategy='majority',n_jobs=n_jobs)
        
        #resampling_pipeline = OneSidedSelection(sampling_strategy='majority',n_jobs=n_jobs, random_state=random_seed)

        X_sample, y_sample = resampling_pipeline.fit_resample(X_reshape, y_train)

        # undersample = RandomUnderSampler(sampling_strategy=strategy, random_state=random_seed)
        # X_sample, y_sample = undersample.fit_resample(X_reshape, y_train)
        X_train = X_sample.reshape(-1, window_size, num_features)
        y_train = y_sample
        Logger.info(f"===> After Undersampling {Counter(y_train)}")

        del X_reshape, X_sample, y_sample
        gc.collect()


    # Scale data
    if cl.config.dataset.scaler_type:
        samples, window_size, num_features = X_train.shape

        Logger.info(f"DL Personalization ===> Scaling dataframes...")
        scaler = dp.get_scaler()

        Logger.info(f"DL Personalization ===> Before Scaling: | X_train mean: {np.mean(X_train.reshape(-1, num_features), axis=1)} | X_train std: {np.std(X_train.reshape(-1, num_features), axis=1)} | X_val mean: {np.mean(X_val.reshape(-1, num_features), axis=1)} | X_val std: {np.std(X_val.reshape(-1, num_features), axis=1)}")
        
        X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(-1, window_size, num_features)
        X_val = scaler.transform(X_val.reshape(-1, num_features)).reshape(-1, window_size, num_features)
        
        # Fit transform with new scaler
        X_infer = scaler.transform(X_infer.reshape(-1, num_features)).reshape(-1, window_size, num_features)
        Logger.info(f"DL Personalization ===> After Scaling: | X_train mean: {np.mean(X_train.reshape(-1, num_features), axis=1)} | X_train std: {np.std(X_train.reshape(-1, num_features), axis=1)} | X_val mean: {np.mean(X_val.reshape(-1, num_features), axis=1)} | X_val std: {np.std(X_val.reshape(-1, num_features), axis=1)}")
    else:
        scaler = None


    # Create datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float())
    infer_dataset = TensorDataset(torch.from_numpy(X_infer), torch.from_numpy(y_infer).float())
    train_loader = dp.load_dataloader(train_dataset, multi_gpu)
    val_loader = dp.load_dataloader(val_dataset, multi_gpu)
    infer_loader = dp.load_dataloader(infer_dataset, multi_gpu, False)
    
    # Load Checkpoint    
    checkpoint = t.load_checkpoint()
    state_checkpoint = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint


    if state_checkpoint is None:
        Logger.info("No checkpoint loaded")


    # Compute weights
    #class_weights = torch.from_numpy(np.array([1.0433, 0.9601])).float()
    class_weights = dp.compute_weights(y_train)
    Logger.info(f"Class weights: {class_weights}")
    print(f"Class weights: {class_weights}")
    class_weights = class_weights.to(device)
    optimizer = t.load_optim(state_checkpoint.best_model, multi_gpu)
    #criterion = t.load_criterion().to(device)
    criterion = t.load_criterion(class_weights).to(device)
    lr_scheduler = t.load_lr_scheduler(optimizer)
    
    # Inference before training
    Logger.info("Inference before training:")
    print("Inference before training:")
    infer_metrics_0 = t.run_epoch(0, "inference", infer_loader, 
                            state_checkpoint.best_model, 
                            criterion,
                            optimizer,
                            lr_scheduler,
                            device, 
                            is_binary=is_binary,
                            threshold= binary_threshold)[0]
    infer_metrics_0.info()
    infer_metrics_0.save_cm(" [Before Personalization]")

    # Save inference metrics
    msg_0 = om.save_object(infer_metrics_0, cl.config.folder, dm.FolderType.results, "inference_metrics_before.pkl" )
    Logger.info(msg_0)

    del X_train, X_val, X_infer, y_train, y_val, y_infer, X_personalized, y_personalized, train_dataset, val_dataset, infer_dataset
    gc.collect()

    # Set the flag to freeze all layers except fc_layers
    freeze_layers = cl.config.checkpoint.freeze_layers

    # Modules to keep trainable
    trainable_module_names = cl.config.checkpoint.trainable_layers #[] #["fc_layers", "transformer_encoder", "cls_token", "positional_embedding"]

    # Loop through the model's parameters and set requires_grad based on module names
    for name, param in state_checkpoint.best_model.named_parameters():
        module_name = name.split(".")[0]
        if module_name in trainable_module_names:
            param.requires_grad = not freeze_layers
        
    # Verify which layers are frozen and which are not
    for name, param in state_checkpoint.best_model.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
        Logger.info(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    # Train Model
    state, state_l = t.train_model(state_checkpoint.best_model, criterion, 
                        optimizer, lr_scheduler,
                        train_loader, val_loader, device, optional_name=f"_personalized",
                        is_binary=is_binary,
                        threshold= cl.config.train.binary_threshold)
    state.info()
    state.scaler = scaler
    state_l.scaler = scaler

    # Inference after training
    Logger.info("Inference after training:")
    print("Inference after training:")
    infer_metrics_1 = t.run_epoch(0, "inference", infer_loader, 
                            state.best_model, 
                            criterion,
                            optimizer,
                            lr_scheduler,
                            device, 
                            is_binary=is_binary,
                            threshold= binary_threshold)[0]
    infer_metrics_1.info()
    infer_metrics_1.save_cm(info=" [After Personalization]")

    Logger.info("======"*20)
    Logger.info("Inference after training with best loss model:")
    infer_metrics_2 = t.run_epoch(0, "inference", infer_loader, 
                            state_l.best_model, 
                            criterion,
                            optimizer,
                            lr_scheduler,
                            device, 
                            is_binary=is_binary,
                            threshold= binary_threshold)[0]
    infer_metrics_2.info()
    infer_metrics_2.save_cm(info=" [BestLoss][After Personalization]")
    
    msg = om.save_object(infer_metrics_1, cl.config.folder, dm.FolderType.results, "inference.pkl" )
    Logger.info(msg)

    msg = om.save_object(infer_metrics_2, cl.config.folder, dm.FolderType.results, "inference_loss.pkl" )
    Logger.info(msg)


    # Visuals
    state.plot_losses(title=f" Personalized on {personalized_subject} | {cl.config.file_name}")
    state.plot_f1_scores(title=f" Personalized on {personalized_subject} | {cl.config.file_name}")
    
    infer_data = []
    ground_truth = []

    for X, y in infer_loader:
        infer_data.append(X.numpy())
        ground_truth.append(y.numpy())
     
    # Concatenate the NumPy arrays to get the final NumPy array with the same batch size
    infer_array = np.concatenate(infer_data, axis=0)
    ground_truth_array = np.concatenate(ground_truth, axis=0)

    # Normalize
    infer_array = dp.scale_arr(infer_array)

    # Check the shape of the resulting NumPy array
    print("Shape of Infer array:", infer_array.shape)

    # Visuals
    #pl.plot_sensor_data(infer_array, ground_truth_array, infer_metrics_0.y_pred, save=False, title=f" Before Personalization | Sub ID:{personalized_subject}", sensor="accx")
    #pl.plot_sensor_data(infer_array, ground_truth_array, infer_metrics_1.y_pred, save=False, title=f" After Personalization | Sub ID:{personalized_subject}", sensor="accx")

    lower = 35
    upper = 45

    pl.plot_sensor_data(infer_array[lower:upper], infer_metrics_0.y_true[lower:upper], infer_metrics_0.y_pred[lower:upper], save=True, title=f" Before Personalization | Sub ID:{personalized_subject}", sensor="acc")
    pl.plot_sensor_data(infer_array[lower:upper], infer_metrics_1.y_true[lower:upper], infer_metrics_1.y_pred[lower:upper], save=True, title=f" After Personalization | Sub ID:{personalized_subject}", sensor="acc")
    pl.plot_sensor_data(infer_array[lower:upper], infer_metrics_2.y_true[lower:upper], infer_metrics_2.y_pred[lower:upper], save=True, title=f" Best Loss | After Personalization | Sub ID:{personalized_subject}", sensor="acc")
    pl.plot_sensor_data(infer_array[lower:upper], infer_metrics_0.y_true[lower:upper], infer_metrics_0.y_pred[lower:upper], save=True, title=f" Before Personalization | Sub ID:{personalized_subject}", sensor="gyro")
    pl.plot_sensor_data(infer_array[lower:upper], infer_metrics_1.y_true[lower:upper], infer_metrics_1.y_pred[lower:upper], save=True, title=f" After Personalization | Sub ID:{personalized_subject}", sensor="gyro")
    pl.plot_sensor_data(infer_array[lower:upper], infer_metrics_2.y_true[lower:upper], infer_metrics_2.y_pred[lower:upper], save=True, title=f" Best Loss | After Personalization | Sub ID:{personalized_subject}", sensor="gyro")
    
    # Trainable parameters
    trainable_params = sum(p.numel() for p in state.best_model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)
    # Total parameters
    total_params = sum(p.numel() for p in state.best_model.parameters())
    print("Total parameters:", total_params)

    Logger.info(f"Trainable parameters: {trainable_params}")
    Logger.info(f"Total parameters: {total_params}")


    print("======"*10)
    print(f"DL Personalization Results [Subject: {cl.config.dataset.personalized_subject}] [Train Ratio: {cl.config.dataset.train_ratio} | Inference Ratio: {cl.config.dataset.inference_ratio}]:")
    print(f"Folder: {cl.config.folder}")
    print(f'''[Based on F1-Score] | Best Epoch: {state.best_epoch}\nTrain F1-Score: {state.best_train_metrics.f1_score:.2f} | Val F1-Score: {state.best_val_metrics.f1_score:.2f}\n
    Before Inference F1-Score: {infer_metrics_0.f1_score:.2f} | After Inference F1-Score: {infer_metrics_1.f1_score:.2f}\n
    Train Loss: {state.best_train_metrics.loss:.2f} | Val Loss: {state.best_val_metrics.loss:.2f}''')
    Logger.info("======"*10)
    Logger.info(f"DL Personalization Results [Subject: {cl.config.dataset.personalized_subject}]:")
    Logger.info(f"Folder: {cl.config.folder}")
    Logger.info(f'''[Based on F1-Score] | Best Epoch: {state.best_epoch}\nTrain F1-Score: {state.best_train_metrics.f1_score:.2f} | Val F1-Score: {state.best_val_metrics.f1_score:.2f}\n
    Before Inference F1-Score: {infer_metrics_0.f1_score:.2f} | After Inference F1-Score: {infer_metrics_1.f1_score:.2f}\n
    Train Loss: {state.best_train_metrics.loss:.2f} | Val Loss: {state.best_val_metrics.loss:.2f}''')
    
    print("++++++++"*10)
    print(f'''[Based on Val Loss] | Best Epoch: {state_l.best_epoch}\nTrain F1-Score: {state_l.best_train_metrics.f1_score:.2f} | Val F1-Score: {state_l.best_val_metrics.f1_score:.2f}\n
    Before Inference F1-Score: {infer_metrics_0.f1_score:.2f} | After Inference F1-Score: {infer_metrics_2.f1_score:.2f}\n
    Train Loss: {state_l.best_train_metrics.loss:.2f} | Val Loss: {state_l.best_val_metrics.loss:.2f}''')
    
    Logger.info("++++++++"*10)
    Logger.info(f'''[Based on Val Loss] | Best Epoch: {state_l.best_epoch}\nTrain F1-Score: {state_l.best_train_metrics.f1_score:.2f} | Val F1-Score: {state_l.best_val_metrics.f1_score:.2f}\n
    Before Inference F1-Score: {infer_metrics_0.f1_score:.2f} | After Inference F1-Score: {infer_metrics_2.f1_score:.2f}\n
    Train Loss: {state_l.best_train_metrics.loss:.2f} | Val Loss: {state_l.best_val_metrics.loss:.2f}''')
    
    

# Function to load data
def load_data():
    shelf_name = cl.config.dataset.name
    random_seed = cl.config.dataset.random_seed
    shuffle = cl.config.dataset.shuffle
    personalized_subject = str(cl.config.dataset.personalized_subject)
    train_ratio = cl.config.dataset.train_ratio
    test_ratio = cl.config.dataset.test_ratio
    inference_ratio = cl.config.dataset.inference_ratio
    remaining_ratio = 1 - (train_ratio + test_ratio + inference_ratio)

    if inference_ratio + test_ratio + train_ratio > 1:
        raise ValueError("Total ratio is greater than 1. Adjust the ratios in config file.")

    # Load python dataset
    X_dict, y_dict = dp.load_shelves(shelf_name, personalized_subject)

    X_personalized = X_dict[personalized_subject]
    y_personalized = y_dict[personalized_subject]

    Logger.info(f"Total samples : | X_personalized shape: {X_personalized.shape} | y_personalized shape: {y_personalized.shape}")

    del X_dict, y_dict
    gc.collect()

    
    #X_infer, X_temp, y_infer, y_temp = train_test_split(X_personalized, y_personalized, train_size= inference_ratio, shuffle=False)

    # Split data
    if shuffle:
        X_infer, X_temp, y_infer, y_temp = train_test_split(X_personalized, y_personalized, train_size= inference_ratio,stratify=y_personalized, shuffle=True, random_state=random_seed)
        X_train, X_temp, y_train, y_temp = train_test_split(X_temp, y_temp, train_size= train_ratio/(train_ratio + test_ratio), stratify = y_temp, shuffle=True, random_state = random_seed)
        if remaining_ratio > 0:
            X_val, X_temp, y_val, y_temp = train_test_split(X_temp, y_temp, train_size= test_ratio/(remaining_ratio + test_ratio), stratify = y_temp, shuffle=True, random_state = random_seed)
        else:
            X_val, y_val = X_temp, y_temp
    else:
        X_infer, X_temp, y_infer, y_temp = train_test_split(X_personalized, y_personalized, train_size= inference_ratio, shuffle=False)
        X_train, X_temp, y_train, y_temp = train_test_split(X_temp, y_temp, train_size= train_ratio/(train_ratio + test_ratio), shuffle=False)
        if remaining_ratio > 0:
            X_val, X_temp, y_val, y_temp = train_test_split(X_temp, y_temp, train_size= test_ratio/(remaining_ratio + test_ratio), shuffle=False)
        else:
            X_val, y_val = X_temp, y_temp

    Logger.info(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    Logger.info(f"X_val shape: {X_val.shape} | y_val shape: {y_val.shape}")
    Logger.info(f"X_infer shape: {X_infer.shape} | y_infer shape: {y_infer.shape}")
    Logger.info(f"X_temp shape: {X_temp.shape} | y_temp shape: {y_temp.shape}")

    return X_train, X_val, X_infer, y_train, y_val, y_infer


def scale_datasets(X_train, X_val, X_infer):
    # Scale data
    if cl.config.dataset.scaler_type:
        samples, window_size, num_features = X_train.shape

        Logger.info(f"DL Personalization ===> Scaling dataframes...")
        scaler = dp.get_scaler()

        Logger.info(f"DL Personalization ===> Before Scaling: | X_train mean: {np.mean(X_train.reshape(-1, num_features), axis=1)} | X_train std: {np.std(X_train.reshape(-1, num_features), axis=1)} | X_val mean: {np.mean(X_val.reshape(-1, num_features), axis=1)} | X_val std: {np.std(X_val.reshape(-1, num_features), axis=1)}")
        
        X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(-1, window_size, num_features)
        X_val = scaler.transform(X_val.reshape(-1, num_features)).reshape(-1, window_size, num_features)
        
        # Fit transform with new scaler
        X_infer = scaler.transform(X_infer.reshape(-1, num_features)).reshape(-1, window_size, num_features)
        Logger.info(f"DL Personalization ===> After Scaling: | X_train mean: {np.mean(X_train.reshape(-1, num_features), axis=1)} | X_train std: {np.std(X_train.reshape(-1, num_features), axis=1)} | X_val mean: {np.mean(X_val.reshape(-1, num_features), axis=1)} | X_val std: {np.std(X_val.reshape(-1, num_features), axis=1)}")
    else:
        scaler = None

    return X_train, X_val, X_infer, scaler


# Function to ensemble models
def ensemble(device, multi_gpu=False):
    print("Device:", device)

    print("======"*20)
    # start
    start = datetime.datetime.now()
    Logger.info(f"Ensemble Personalization Start time: {start}")
    print(f"Ensemble Personalization Start time: {start}")

    is_binary = cl.config.dataset.num_classes < 3
    random_seed = cl.config.dataset.random_seed
    shuffle = cl.config.dataset.shuffle
    personalized_subject = str(cl.config.dataset.personalized_subject)
    binary_threshold = cl.config.train.binary_threshold
    
    # Load datasets
    X_train, X_val, X_infer, y_train, y_val, y_infer = load_data()

    # Scale datasets
    X_train, X_val, X_infer, scaler = scale_datasets(X_train, X_val, X_infer)

    # Create datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float())
    infer_dataset = TensorDataset(torch.from_numpy(X_infer), torch.from_numpy(y_infer).float())
    train_loader = dp.load_dataloader(train_dataset, multi_gpu)
    val_loader = dp.load_dataloader(val_dataset, multi_gpu)
    infer_loader = dp.load_dataloader(infer_dataset, multi_gpu, False)


    # Load functions
    class_weights = dp.compute_weights(y_train)
    class_weights = class_weights.to(device)
    criterion = t.load_criterion(class_weights).to(device)
    
    # Delete variables
    del X_train, X_val, X_infer, y_train, y_val, y_infer, train_dataset, val_dataset, infer_dataset
    gc.collect()

    #optimizer = t.load_optim(state_checkpoint.best_model, multi_gpu)
    #criterion = t.load_criterion().to(device)
    #lr_scheduler = t.load_lr_scheduler(optimizer)

    # Load Checkpoint
    checkpoint_folder = cl.config.train.checkpoint.split('/')[0]
    models_paths = dm.get_best_models(path = os.path.join(cl.config.models_folder, checkpoint_folder))
    print("Best Models:", models_paths)

    models = {}

    for sub, path in models_paths.items():
        models[sub] = t.load_checkpoint(path).best_model
    
    Logger.info("======"*20)
    # Inference before training
    Logger.info("Inference before training:")
    print("Inference before training:")
    infer_metrics_0 = run_ensemble(infer_loader, models, criterion, device, is_binary, binary_threshold)

    infer_metrics_0.info()
    infer_metrics_0.save_cm(" [Ensemble Before Personalization]")
    Logger.info("======"*20)


    # Train Models
    states = train_ensemble(train_loader, val_loader, scaler, models, criterion, device, is_binary, binary_threshold, multi_gpu)
    
    new_models = {}

    for sub, state in states.items():
        new_models[sub] = state.best_model

    Logger.info("======"*20)
    # Inference after training
    Logger.info("Inference after training:")
    print("Inference after training:")
    infer_metrics_1 = run_ensemble(infer_loader, new_models, criterion, device, is_binary, binary_threshold)
    
    infer_metrics_1.info()
    infer_metrics_1.save_cm(" [Ensemble After Personalization]")

    Logger.info("======"*20)

    # Visuals
    infer_data = []
    ground_truth = []

    for X, y in infer_loader:
        infer_data.append(X.numpy())
        ground_truth.append(y.numpy())
     
    # Concatenate the NumPy arrays to get the final NumPy array with the same batch size
    infer_array = np.concatenate(infer_data, axis=0)
    ground_truth_array = np.concatenate(ground_truth, axis=0)
    
    # Check the shape of the resulting NumPy array
    print("Shape of Infer array:", infer_array.shape)
    Logger.info(f"Shape of Infer array: {infer_array.shape}")

    # Normalize
    #infer_array = dp.scale_arr(infer_array)

    lower = 22
    upper = 42

    pl.plot_sensor_data(infer_array[lower:upper], ground_truth_array[lower:upper], infer_metrics_0.y_pred[lower:upper], save=True, title=f" Ensemble Before Personalization | Sub ID:{personalized_subject}", sensor="acc")
    pl.plot_sensor_data(infer_array[lower:upper], ground_truth_array[lower:upper], infer_metrics_1.y_pred[lower:upper], save=True, title=f" Ensemble After Personalization | Sub ID:{personalized_subject}", sensor="acc")

    pl.plot_sensor_data(infer_array[lower:upper], ground_truth_array[lower:upper], infer_metrics_0.y_pred[lower:upper], save=True, title=f" Ensemble Before Personalization | Sub ID:{personalized_subject}", sensor="gyro")
    pl.plot_sensor_data(infer_array[lower:upper], ground_truth_array[lower:upper], infer_metrics_1.y_pred[lower:upper], save=True, title=f" Ensemble After Personalization | Sub ID:{personalized_subject}", sensor="gyro")

    model = new_models['1']
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    Logger.info(f"Trainable parameters: {trainable_params}")
    Logger.info(f"Total parameters: {total_params}")

    end = datetime.datetime.now()
    duration = end - start
    Logger.info(f"Personalization End time: {end} | Duration: {duration}")

# Function to train models
def train_ensemble(train_loader, val_loader, scaler, models, criterion, device, is_binary, binary_threshold, multi_gpu=False):
    Logger.info("Training Ensemble...")
    personalized_subject = str(cl.config.dataset.personalized_subject)

    states = {}

    Logger.info("======"*20)

    # Loop over models
    for fold, model in models.items():
        Logger.info(f"Training on Fold: {fold} k-fold:")
        # Load optimizer and lr_scheduler
        optimizer = t.load_optim(model, multi_gpu)
        lr_scheduler = t.load_lr_scheduler(optimizer)

        # Training
        state = t.train_model(model, criterion, 
                        optimizer, lr_scheduler,
                        train_loader, val_loader, device, optional_name=f"_cv-{fold}_personalized",
                        is_binary=is_binary,
                        threshold= binary_threshold)
        state.scaler = scaler
        state.plot_losses(title=f" {fold}_k-fold Personalized on {personalized_subject}")
        state.plot_f1_scores(title=f" {fold}_k-fold Personalized on {personalized_subject}")

        # Append
        states[fold] = state
        Logger.info("Training on Fold: {fold}, is completed.")
        Logger.info("======"*20)
    # Trained states
    return states



def run_ensemble(infer_loader, models, criterion, device, is_binary, binary_threshold, multi_gpu=False):

    metrices = []
    # Loop over models
    for fold, model in models.items():
        Logger.info(f"Inference Output of {fold}_k-fold:")

        # Load optimizer and lr_scheduler
        optimizer = t.load_optim(model, multi_gpu)
        lr_scheduler = t.load_lr_scheduler(optimizer)

        # Inference
        infer_metrics = t.run_epoch(0, "inference", infer_loader, 
                            model, 
                            criterion,
                            optimizer,
                            lr_scheduler,
                            device, 
                            is_binary=is_binary,
                            threshold= binary_threshold)[0]
        
        infer_metrics.info()
        infer_metrics.save_cm(info=f" [Ensemble k-Fold: {fold}]")
        metrices.append(infer_metrics)
    

    # Ensemble outputs
    outputs = [metric.y_pred for metric in metrices]

    # Majority voting
    ensemble_output = ensemble_majority_voting(outputs)

    # Create Ensemble metrics
    ensemble_metrics = Metrics(0, is_binary)
    ensemble_metrics.y_true = metrices[0].y_true
    ensemble_metrics.y_pred = ensemble_output
    ensemble_metrics.calculate_metrics()
    Logger.info("Ensemble Output:")
    ensemble_metrics.info()
    ensemble_metrics.save_cm(info=" [Ensemble Output]")

    return ensemble_metrics    


def ensemble_majority_voting(predictions:list):
  
    # 2D numpy array (rows = samples, columns = models)
    stacked_predictions = np.vstack(predictions)
    
    # Calculate the most common prediction for each rows
    ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked_predictions)

    return ensemble_predictions