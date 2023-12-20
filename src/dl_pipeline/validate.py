import datetime
import gc
import numpy as np
import random
from collections import Counter
import copy

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
from src.helper.data_model import TaskType
from imblearn.under_sampling import OneSidedSelection, NearMiss, RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split

def get_mean_scores(states:[State], phase:str):
    """Method to get mean scores from list of states.

    Args:
        states ([State]): List of states.

    Returns:
        [type]: [description]
    """
    f1_scores = []
    recall_scores = []
    precision_scores = []
    specificity_scores = []
    jaccard_scores = []
    accuracy_scores = []


    for state in states:
        if isinstance(state, int):
            continue
        metric = state.best_train_metrics if phase == "train" else state.best_val_metrics

        f1_scores.append(metric.f1_score)
        recall_scores.append(metric.recall_score)
        precision_scores.append(metric.precision_score)
        specificity_scores.append(metric.specificity_score)
        jaccard_scores.append(metric.jaccard_score)
        accuracy_scores.append(metric.accuracy)
    
    mean_scores = { "f1_score": np.mean(f1_scores),
                    "recall_score": np.mean(recall_scores), 
                    "precision_score": np.mean(precision_scores),
                    "specificity_score": np.mean(specificity_scores),
                    "jaccard_score": np.mean(jaccard_scores),
                    "accuracy": np.mean(accuracy_scores)
                    }

    return mean_scores


def stratified_k_fold_cv(device, multi_gpu=False):
    print("======"*5)
    # start
    start = datetime.datetime.now()
    Logger.info(f"Stratified Cross-Validation Start time: {start}")
    print(f"Stratified Cross-Validation Start time: {start}")

    # Empty dict to store results
    results = {}
    best_val_loss = np.inf
    best_f1_score = 0.0
    best_fold = None
    best_fold_l = None
    #is_binary = cl.config.dataset.num_classes < 3
    is_binary = cl.config.train.task_type.value != TaskType.Multiclass_classification.value
    
    shelf_name = cl.config.dataset.name
    random_seed = cl.config.dataset.random_seed
    n_splits = cl.config.train.cross_validation.k_folds
    shuffle = cl.config.dataset.shuffle
    new_seed = random_seed if shuffle else None
    stratified_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=new_seed if shuffle else None)
    train_ratio = cl.config.dataset.train_ratio

    # Load python dataset
    X_dict, y_dict = dp.load_shelves(shelf_name)

    # All subjects
    subjects = list(X_dict.keys())
    
    # Remove personalized subject
    personalized_subject = str(cl.config.dataset.personalized_subject)
    
    if personalized_subject in subjects:
        Logger.info(f"Removing personalized subject from dataset...:{personalized_subject}")
        print(f"Removing personalized subject:{personalized_subject}")
        subjects.remove(personalized_subject)
    print("Subjects applied:", subjects)

    X_all =  np.concatenate([X_dict[subject] for subject in subjects], axis=0)
    y_all =  np.concatenate([y_dict[subject] for subject in subjects], axis=0)
    X_personalize = X_dict[personalized_subject]
    y_personalize = y_dict[personalized_subject]

    if cl.config.dataset.sensor == "acc":
       X_all = X_all[:, :, :3]
       Logger.info("Using accelerometer data only.")
    elif cl.config.dataset.sensor == "gyro":
       X_all = X_all[:, :, 3:]
       Logger.info("Using gyroscope data only.")
    
    Logger.info(f"Total data shape: {X_all.shape} | Total labels shape: {y_all.shape}")

    # Reshape
    num, window_size, num_features = X_all.shape
    X_all = X_all.reshape(num, -1)

    # Split data
    if shuffle:
        X_train, X_inference, y_train, y_inference = train_test_split(X_all, y_all, train_size = train_ratio, stratify = y_all, shuffle=shuffle, random_state = new_seed)
    else:
        X_train, X_inference, y_train, y_inference = train_test_split(X_all, y_all, train_size = train_ratio)

    X_inference = X_inference.reshape(-1, window_size, num_features)

    Logger.info(f"Total Train size: {len(X_train)} | Counts: {Counter(y_train)}")
    Logger.info(f"Inference size: {len(X_inference)} | Counts: {Counter(y_inference)}")

    del X_dict, y_dict
    gc.collect()

    # Loop through splits
    for i, (train_index, val_index) in enumerate(stratified_kf.split(X_train, y_train)):
        # Split data
        train_data = X_train[train_index].reshape(-1, window_size, num_features)
        train_labels = y_train[train_index]
        val_data = X_train[val_index].reshape(-1, window_size, num_features)
        val_labels = y_train[val_index]

        # start of k-fold
        k_start = datetime.datetime.now()
        Logger.info(f"-------------------")
        Logger.info(f"\nStarting Stratified_k-fold cross-validation on fold no. {i+1} start time: {k_start}...")
        
        # Info
        Logger.info(f"Stratified_k-Fold:{i+1} ===> Train data shape: {train_data.shape} | Train labels shape: {train_labels.shape}")
        Logger.info(f"Stratified_k-Fold:{i+1} ===> Val data shape: {val_data.shape} | Val labels shape: {val_labels.shape}") 


        # Sampling
        if cl.config.dataset.sampling:
            samples, window_size, num_features = train_data.shape
            Logger.info(f"Stratified_k-Fold:{i+1} ===> Before Undersampling {Counter(train_labels)}")
            X_reshape = train_data.reshape(samples, -1)       
            #null_samples = int(np.sqrt(samples))
            #k = null_samples if (null_samples%2 == 1) else (null_samples-1)
            k = 7
            #undersample = OneSidedSelection(n_neighbors=k, sampling_strategy='majority', n_jobs=-1, random_state=new_seed)
            counts = Counter(train_labels.astype(int))
            max_count = max(counts.values())
            print(f"Before sampling: {Counter(train_labels.astype(int))}")
            under_sampling_strategy = {
                0: int(max_count*cl.config.dataset.alpha)
            }
            print("Sampling strategy:", under_sampling_strategy)

            undersample = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=new_seed)
            X_sample, y_sample = undersample.fit_resample(X_reshape, train_labels)
            train_data = X_sample.reshape(-1, window_size, num_features)
            train_labels = y_sample
            Logger.info(f"Stratified_k-Fold:{i+1} ===> After Undersampling {Counter(train_labels)}")
            
            del X_reshape, X_sample, y_sample, undersample
            gc.collect()

        Logger.info(f"Stratified_k-Fold:{i+1} ===> X_train shape: {train_data.shape} | y_train shape: {train_labels.shape}")
        Logger.info(f"Stratified_k-Fold:{i+1} ===> X_val shape: {val_data.shape} | y_val shape: {val_labels.shape}")

        # Scale dataframes
        if cl.config.dataset.scaler_type:
            samples, window_size, num_features = train_data.shape

            Logger.info(f"Stratified_k-Fold:{i+1} ===> Scaling dataframes...")
            scaler = dp.get_scaler()

            Logger.info(f"Stratified_k-Fold:{i+1} ===> Before Scaling: | X_train mean: {np.mean(train_data.reshape(-1, num_features), axis=1)} | X_train std: {np.std(train_data.reshape(-1, num_features), axis=1)} | X_val mean: {np.mean(val_data.reshape(-1, num_features), axis=1)} | X_val std: {np.std(val_data.reshape(-1, num_features), axis=1)}")
            
            train_data = scaler.fit_transform(train_data.reshape(-1, num_features)).reshape(-1, window_size, num_features)
            val_data = scaler.transform(val_data.reshape(-1, num_features)).reshape(-1, window_size, num_features)
            
            Logger.info(f"Stratified_k-Fold:{i+1} ===> After Scaling: | X_train mean: {np.mean(train_data.reshape(-1, num_features), axis=1)} | X_train std: {np.std(train_data.reshape(-1, num_features), axis=1)} | X_val mean: {np.mean(val_data.reshape(-1, num_features), axis=1)} | X_val std: {np.std(val_data.reshape(-1, num_features), axis=1)}")
        else:
            scaler = None

        # Create datasets
        train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels).float())
        val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_labels).float())
    
        Logger.info(f"Stratified_k-Fold:{i+1} ===> Train dataset size: {len(train_dataset)} | Val dataset size: {len(val_dataset)} | Sample shape: {train_dataset[0][0].shape}")

        
        
        # Create data loaders
        Logger.info(f"Stratified_k-Fold:{i+1} ===> Creating dataloaders...")
        train_loader = dp.load_dataloader(train_dataset, multi_gpu)
        val_loader = dp.load_dataloader(val_dataset, multi_gpu)
        

        # Compute weights
        Logger.info(f"Stratified_k-Fold:{i+1} ===> Computing class weights...")   
        class_weights = dp.compute_weights(train_labels)
        class_weights = class_weights.to(device)

        # Del 
        del train_data, val_data, train_labels, val_labels
        gc.collect()

        # Train model
        Logger.info(f"Stratified_k-Fold:{i+1} ===> Loading model...")
        # Load Traning parameters
        model = t.load_network(multi_gpu)
        model = model.to(device)
        print("Model loadded on device: ", device)
        print("Multiple GPUs: ", multi_gpu)

        optimizer = t.load_optim(model, multi_gpu)
        criterion = t.load_criterion(class_weights)
        lr_scheduler = t.load_lr_scheduler(optimizer)

        Logger.info(f"Stratified_k-Fold:{i+1} ===> Training model...")

        # Train Model
        state, state_l = t.train_model(model, criterion, 
                            optimizer, lr_scheduler,
                            train_loader, val_loader, device, optional_name=f"_cv-{i+1}_stratified_fold",
                            is_binary=is_binary,
                            threshold= cl.config.train.binary_threshold)

        if state.best_val_metrics.f1_score > best_f1_score:
            best_f1_score = state.best_val_metrics.f1_score
            best_fold = i
            cl.config.checkpoint.filename = state.file_name
            Logger.info(f"Check F1 Score: Stratified_k-Fold:{i+1} ===> New best fold: {best_fold +1} with validation Loss: {state.best_val_metrics.loss} | F1-score:{best_f1_score} | Best Epoch: {state.best_epoch}")


        if state.best_val_metrics.loss < best_val_loss:
            best_val_loss = state.best_val_metrics.loss
            best_fold_l = i
            Logger.info(f"Check Val Loss: Stratified_k-Fold:{i+1} ===> New best fold: {best_fold_l +1} with validation Loss: {best_val_loss} | F1-score:{state.best_val_metrics.f1_score} | Best Epoch: {state.best_epoch}")


        state.info()
        state.scaler = scaler

        state_l.scaler = scaler

        # Visuals
        state.plot_losses(title=f" Stratified CV | k-Fold: {i+1} | {cl.config.file_name}")
        state.plot_f1_scores(title=f" Stratified CV | k-Fold: {i+1} | {cl.config.file_name}")

        # Save state to dict
        results[i] = {}
        results[i]['f1'] = copy.deepcopy(state)
        results[i]['loss'] = copy.deepcopy(state_l)

        del state, state_l
        gc.collect()

        # End of k-fold
        k_end = datetime.datetime.now()
        Logger.info(f"Stratified_k-Fold:{i+1} ===> End of k-fold cross-validation on fold no. {i+1} end time: {k_end} | Duration: {k_end - k_start}")
        print(f"Stratified_k-Fold:{i+1} ===> End of k-fold cross-validation on fold no. {i+1} end time: {k_end} | Duration: {k_end - k_start}")

    #Logger.info(f"Best Stratified_k-Fold: {best_fold+1} with validation Loss: {best_val_loss}")
    Logger.info(f"Best Stratified_k-Fold: {best_fold+1} with Val F1-score: {best_f1_score}")

    # Save metrics
    Logger.info("Saving results...")
    results['best_fold'] = best_fold + 1
    results['best_fold_l'] = best_fold_l + 1

    msg = om.save_object(results, cl.config.folder, dm.FolderType.results, "results.pkl" )
    Logger.info(msg)

    # Info
    Logger.info("Average Scores:")
    
    # TO DO: Add option to use warn metrics
    #use_warn_score = cl.config.metrics.use_warn_metrics

    states = [v['f1'] for k, v in results.items() if not isinstance(v, int)]

    Logger.info(f"Training Average-Scores: {get_mean_scores(states, 'train' )}")
    Logger.info(f"Validation Average-Scores: {get_mean_scores(states, 'val')}")

    # End LOSOCV
    end_train = datetime.datetime.now()
    Logger.info(f"Stratified Cross-Validation End time: {end_train}")
    Logger.info(f"Stratified Cross-Validation Duration: {end_train - start}")

    ###############################################
    ################ Inference ####################
    ###############################################
    Logger.info("Inference started...")
    print("======"*10)
    print("Inference started...")

    best_state = results[best_fold]['f1']
    best_state_l = results[best_fold_l]['loss']

    # Scale
    if best_state.scaler:
        X_in = best_state.scaler.transform(X_inference.reshape(-1, num_features)).reshape(-1, window_size, num_features)
        X_p = best_state.scaler.transform(X_personalize.reshape(-1, num_features)).reshape(-1, window_size, num_features)
    
    inference_dataset = TensorDataset(torch.from_numpy(X_in).float(), torch.from_numpy(y_inference).float())
    personalize_dataset = TensorDataset(torch.from_numpy(X_p).float(), torch.from_numpy(y_personalize).float())

    Logger.info(f"Inference dataset size: {len(inference_dataset)} | Sample shape: {inference_dataset[0][0].shape}")
    Logger.info(f"Personalize dataset size: {len(personalize_dataset)} | Sample shape: {personalize_dataset[0][0].shape}")

    
    Logger.info("Creating Inference Dataloader...")
    
    # Inference dataloader
    inference_loader = dp.load_dataloader(inference_dataset)
    personalize_loader = dp.load_dataloader(personalize_dataset)

    # Load best criterion
    loss_fn = t.load_criterion(best_state.best_criterion_weight)

    # Inference
    inference_metrics = t.run_epoch(0,"inference", inference_loader,
                                    best_state.best_model,loss_fn,
                                    best_state.best_optimizer, best_state.best_lr_scheduler,
                                    device=device, is_binary= is_binary, threshold=cl.config.train.binary_threshold)[0]

    inference_metrics.info()


    # Save inference metrics
    msg = om.save_object(inference_metrics, cl.config.folder, dm.FolderType.results, "inference_metrics.pkl" )
    Logger.info(msg)
    
    Logger.info("Inference on Personalized Subject : {personalized_subject}")
    print("Inference on Personalized Subject : {personalized_subject}")
    
    inference_metrics_personalize = t.run_epoch(0,"inference", personalize_loader, best_state.best_model,loss_fn,
                                    best_state.best_optimizer, best_state.best_lr_scheduler,    
                                    device=device, is_binary= is_binary, threshold=cl.config.train.binary_threshold)[0]
    inference_metrics_personalize.info()
    
    # Save inference metrics
    msg = om.save_object(inference_metrics_personalize, cl.config.folder, dm.FolderType.results, "inference_metrics_personalize.pkl" )
    Logger.info(msg)

    
    infer_data = [X.numpy() for X, y in inference_loader]

    # Concatenate the NumPy arrays to get the final NumPy array with the same batch size
    infer_array = np.concatenate(infer_data, axis=0)

    # Check the shape of the resulting NumPy array
    print("Shape of Infer array:", infer_array.shape)

    # Visuals
    #pl.plot_sensor_data(infer_array, inferece_metrics.y_true, inferece_metrics.y_pred, save=True, title=f"Inference Result")

    lower = 20
    upper = 30
    #pl.plot_sensor_data(infer_array[lower:upper], inferece_metrics.y_true[lower:upper], inferece_metrics.y_pred[lower:upper], save=True, title=f"Inference Result")
    pl.plot_sensor_data(infer_array[lower:upper], inference_metrics.y_true[lower:upper], inference_metrics.y_pred[lower:upper], save=True, title=f"Inference Results", sensor="acc")
    pl.plot_sensor_data(infer_array[lower:upper], inference_metrics.y_true[lower:upper], inference_metrics.y_pred[lower:upper], save=True, title=f"Inference Results", sensor="gyro")

    # Inferece duration
    end_inference = datetime.datetime.now()
    Logger.info(f"Inference End time: {end_inference}")
    Logger.info(f"Inference Duration: {end_inference - end_train}")
    print(f"Inference End time: {end_inference}")
    print(f"Inference Duration: {end_inference - end_train}")

     ###############################################
    # Comparing best fold with best val loss
    
    if best_state_l.scaler:
        X_in = best_state_l.scaler.transform(X_inference.reshape(-1, num_features)).reshape(-1, window_size, num_features)
    
    inference_dataset = TensorDataset(torch.from_numpy(X_in).float(), torch.from_numpy(y_inference).float())
    inference_loader =  dp.load_dataloader(inference_dataset)
    loss_fn = t.load_criterion(best_state_l.best_criterion_weight)
    inference_metrics_l = t.run_epoch(0,"inference [Loss]", inference_loader,
                                    best_state_l.best_model,loss_fn,   
                                    best_state_l.best_optimizer, best_state_l.best_lr_scheduler,
                                    device=device, is_binary= is_binary, threshold=cl.config.train.binary_threshold)[0]

    Logger.info("Based on Val Loss:")
    inference_metrics_l.info()
    # Save inference metrics
    msg = om.save_object(inference_metrics_l, cl.config.folder, dm.FolderType.results, "inference_metrics_loss.pkl" )
    Logger.info(msg)
    del X_in, X_inference, y_inference, inference_dataset, inference_loader, loss_fn
    gc.collect()
    Logger.info("======"*10)
    print(f"Stratified CV Results [Subject: {cl.config.dataset.personalized_subject}]:")
    Logger.info(f"Stratified CV Results [Subject: {cl.config.dataset.personalized_subject}]:")
    # Results
    Logger.info("Results:")
    Logger.info(f"Folder: {cl.config.folder}")
    cl.config.checkpoint.folder = cl.config.folder
    cl.config.checkpoint.name = best_state.file_name
    Logger.info(f"Architecture: {best_state.best_model.__class__.__name__}")
    print(f"Architecture: {best_state.best_model.__class__.__name__}")
    Logger.info(f"[Based on F1-Score] Best k-Fold: {best_fold+1} | Best Epoch: {best_state.best_epoch}\nTrain F1-Score: {best_state.best_train_metrics.f1_score:.2f} | Val F1-Score: {best_state.best_val_metrics.f1_score:.2f} | Inference F1-Score: {inference_metrics.f1_score:.2f}\nTrain Loss: {best_state.best_train_metrics.loss:.2f} | Val Loss: {best_state.best_val_metrics.loss:.2f}")
    Logger.info("++++++++"*10)
    Logger.info(f"[Based on Loss] Best k-Fold: {best_fold_l+1} | Best Epoch: {best_state_l.best_epoch}\nTrain F1-Score: {best_state_l.best_train_metrics.f1_score:.2f} | Val F1-Score: {best_state_l.best_val_metrics.f1_score:.2f} | Inference F1-Score: {inference_metrics_l.f1_score:.2f}\nTrain Loss: {best_state_l.best_train_metrics.loss:.2f} | Val Loss: {best_state_l.best_val_metrics.loss:.2f}")

    print("======"*10)
    print("Results:")
    print(f"Folder: {cl.config.folder}")
    print(f'''[Based on F1-Score] Best k-Fold: {best_fold+1} | Best Epoch: {best_state.best_epoch}\n
    Train F1-Score: {best_state.best_train_metrics.f1_score:.2f} | Val F1-Score: {best_state.best_val_metrics.f1_score:.2f} | Inference F1-Score: {inference_metrics.f1_score:.2f}\n
    Train Loss: {best_state.best_train_metrics.loss:.2f} | Val Loss: {best_state.best_val_metrics.loss:.2f}\n
    Inference on Personalized Subject F1-Score: {inference_metrics_personalize.f1_score:.2f}
    ''')
    
    print("++++++++"*10)
    print(f"[Based on Loss] Best k-Fold: {best_fold_l+1} | Best Epoch: {best_state_l.best_epoch}\nTrain F1-Score: {best_state_l.best_train_metrics.f1_score:.2f} | Val F1-Score: {best_state_l.best_val_metrics.f1_score:.2f} | Inference F1-Score: {inference_metrics_l.f1_score:.2f}\nTrain Loss: {best_state_l.best_train_metrics.loss:.2f} | Val Loss: {best_state_l.best_val_metrics.loss:.2f}")

    
# Function to run subjectwise k-fold cross-validation
def subwise_k_fold_cv(device, multi_gpu=False):
    print("Device:", device)

    print("======"*5)
    # start
    start = datetime.datetime.now()
    Logger.info(f"Cross-Validation Start time: {start}")
    print(f"Cross-Validation Start time: {start}")

    # Empty dict to store results
    results = {}
    best_val_loss = np.inf
    best_f1_score = 0.0
    best_fold = None
    is_binary = cl.config.train.task_type.value != TaskType.Multiclass_classification.value
    shelf_name = cl.config.dataset.name
    random_seed = cl.config.dataset.random_seed
    shuffle = cl.config.dataset.shuffle
    new_seed = random_seed if shuffle else None

    # Load python dataset
    X_dict, y_dict = dp.load_shelves(shelf_name)

    # All subjects
    subjects = list(X_dict.keys())

    # Split train and inference subjects
    train_subjects, inference_subjects = dp.split_subjects(subjects)
    Logger.info(f"Train subjects: {train_subjects}")
    Logger.info(f"Inference subjects: {inference_subjects}")

    # Prepare k-fold lists
    if cl.config.train.cross_validation.name == "losocv":
        k_folds = ds.divide_into_groups(train_subjects, len(train_subjects))
    else:
        k_folds = ds.divide_into_groups(train_subjects, cl.config.train.cross_validation.k_folds)

    print(f"Train subjects: {train_subjects}")
    print(f"Inference subjects: {inference_subjects}")

    # Loop through folds
    for i, fold_val_subjects in enumerate(k_folds):
        # start of k-fold
        k_start = datetime.datetime.now()
        Logger.info(f"-------------------")
        Logger.info(f"\nStarting k-fold cross-validation on fold no. {i+1} start time: {k_start}...")
        
        # k-fold train subjects
        fold_train_subjects = list(set(train_subjects) - set(fold_val_subjects)) 
        
        Logger.info(f"k-Fold:{i+1} ===> Validation subjects: {fold_val_subjects} | List size: {len(fold_val_subjects)}")
        Logger.info(f"k-Fold:{i+1} ===> Training subjects: {fold_train_subjects} | List size: {len(fold_train_subjects)}")
        
        print(f"\nk-Fold:{i+1} ===> Validation subjects: {fold_val_subjects} | List size: {len(fold_val_subjects)}")
        print(f"k-Fold:{i+1} ===> Training subjects: {fold_train_subjects} | List size: {len(fold_train_subjects)}")

        # Load numpy datasets
        X_train = np.concatenate([X_dict[subject] for subject in fold_train_subjects], axis=0)
        y_train = np.concatenate([y_dict[subject] for subject in fold_train_subjects], axis=0)

        X_val = np.concatenate([X_dict[subject] for subject in fold_val_subjects], axis=0)
        y_val = np.concatenate([y_dict[subject] for subject in fold_val_subjects], axis=0)
        

        # Check sensor type 
        if cl.config.dataset.sensor == "acc":
            X_train = X_train[:, :, :3]
            X_val = X_val[:, :, :3]
            Logger.info("Using accelerometer data only.")
        elif cl.config.dataset.sensor == "gyro":
            X_train = X_train[:, :, 3:]
            X_val = X_val[:, :, 3:]
            Logger.info("Using gyroscope data only.")
        
        Logger.info(f"k-Fold:{i+1} ===> X_train shape: {X_train.shape} | X_val shape: {X_val.shape}")

        # Sampling
        if cl.config.dataset.sampling:
            samples, window_size, num_features = X_train.shape
            Logger.info(f"k-Fold:{i+1} ===> Before Undersampling {Counter(y_train)}")
            X_reshape = X_train.reshape(samples, -1)       
            #null_samples = int(np.sqrt(samples))
            #k = null_samples if (null_samples%2 == 1) else (null_samples-1)
            k = 7
            #undersample = OneSidedSelection(n_neighbors=k, sampling_strategy='majority', n_jobs=-1, random_state=new_seed)
            counts = Counter(y_train.astype(int))
            max_count = max(counts.values())
            print(f"Before sampling: {Counter(y_train.astype(int))}")
            under_sampling_strategy = {
                0: int(max_count*cl.config.dataset.alpha)
            }
            print("Sampling strategy:", under_sampling_strategy)

            undersample = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=new_seed)
            X_sample, y_sample = undersample.fit_resample(X_reshape, y_train)
            X_train = X_sample.reshape(-1, window_size, num_features)
            y_train = y_sample
            Logger.info(f"k-Fold:{i+1} ===> After Undersampling {Counter(y_train)}")
            
            del X_reshape, X_sample, y_sample, undersample
            gc.collect()

        Logger.info(f"k-Fold:{i+1} ===> X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
        Logger.info(f"k-Fold:{i+1} ===> X_val shape: {X_val.shape} | y_val shape: {y_val.shape}")

        # Scale dataframes
        if cl.config.dataset.scaler_type:
            samples, window_size, num_features = X_train.shape

            Logger.info(f"k-Fold:{i+1} ===> Scaling dataframes...")
            scaler = dp.get_scaler()

            Logger.info(f"k-Fold:{i+1} ===> Before Scaling: | X_train mean: {np.mean(X_train.reshape(-1, num_features), axis=1)} | X_train std: {np.std(X_train.reshape(-1, num_features), axis=1)} | X_val mean: {np.mean(X_val.reshape(-1, num_features), axis=1)} | X_val std: {np.std(X_val.reshape(-1, num_features), axis=1)}")
            
            X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(-1, window_size, num_features)
            X_val = scaler.transform(X_val.reshape(-1, num_features)).reshape(-1, window_size, num_features)
            
            Logger.info(f"k-Fold:{i+1} ===> After Scaling: | X_train mean: {np.mean(X_train.reshape(-1, num_features), axis=1)} | X_train std: {np.std(X_train.reshape(-1, num_features), axis=1)} | X_val mean: {np.mean(X_val.reshape(-1, num_features), axis=1)} | X_val std: {np.std(X_val.reshape(-1, num_features), axis=1)}")
        else:
            scaler = None

        # Create datasets
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float())
    
        Logger.info(f"k-Fold:{i+1} ===> Train dataset size: {len(train_dataset)} | Val dataset size: {len(val_dataset)} | Sample shape: {train_dataset[0][0].shape}")

        
        
        # Create data loaders
        Logger.info(f"k-Fold:{i+1} ===> Creating dataloaders...")
        train_loader = dp.load_dataloader(train_dataset)
        val_loader = dp.load_dataloader(val_dataset)
        

        # Compute weights
        Logger.info(f"k-Fold:{i+1} ===> Computing class weights...")   
        class_weights = dp.compute_weights(y_train)
        class_weights = class_weights.to(device)

        # Del 
        del X_train, X_val, y_train, y_val
        gc.collect()

        # Train model
        Logger.info(f"k-Fold:{i+1} ===> Loading model...")
        # Load Traning parameters
        model = t.load_network(multi_gpu)
        model = model.to(device)
        print("Model loaded on device: ", device)
        print("Multiple GPUs: ", multi_gpu) 
        optimizer = t.load_optim(model)
        criterion = t.load_criterion(class_weights)
        lr_scheduler = t.load_lr_scheduler(optimizer)

        Logger.info(f"k-Fold:{i+1} ===> Training model...")

        # Train Model
        state, state_l = t.train_model(model, criterion, 
                            optimizer, lr_scheduler,
                            train_loader, val_loader, device, optional_name=f"_cv-{i+1}_fold",
                            is_binary=is_binary,
                            threshold= cl.config.train.binary_threshold)
        
        if state.best_val_metrics.f1_score > best_f1_score:
            best_f1_score = state.best_val_metrics.f1_score
            best_fold = i
            Logger.info(f"Check F1 Score: k-Fold:{i+1} ===> New best fold: {best_fold +1} with validation Loss: {state.best_val_metrics.loss} | F1-score:{best_f1_score} | Best Epoch: {state.best_epoch}")

        if state.best_val_metrics.loss < best_val_loss:
            best_val_loss = state.best_val_metrics.loss
            #best_fold = i
            Logger.info(f"Check Val Loss: k-Fold:{i+1} ===> New best fold: {best_fold +1} with validation Loss: {best_val_loss} | F1-score:{state.best_val_metrics.f1_score} | Best Epoch: {state.best_epoch}")

        
        state.info()
        state.scaler = scaler
        state.best_fold = best_fold + 1

        state_l.scaler = scaler
        state_l.best_fold = best_fold + 1

        # Visuals
        state.plot_losses(title=f" CV on k-Fold: {i+1} {cl.config.file_name}")
        state.plot_f1_scores(title=f" CV on k-Fold: {i+1} {cl.config.file_name}")

        # Save state to dict
        results[i]['f1'] = copy.deepcopy(state)
        results[i]['loss'] = copy.deepcopy(state_l)
        
        del state, state_l
        gc.collect()
        
        # End of k-fold
        k_end = datetime.datetime.now()
        Logger.info(f"k-Fold:{i+1} ===> End of k-fold cross-validation on fold no. {i+1} end time: {k_end} | Duration: {k_end - k_start}")
        print(f"k-Fold:{i+1} ===> End of k-fold cross-validation on fold no. {i+1} end time: {k_end} | Duration: {k_end - k_start}")

    #Logger.info(f"Best k-Fold: {best_fold+1} with validation Loss: {best_val_loss}")
    Logger.info(f"Best k-Fold: {best_fold+1} | F1-score: {best_f1_score} | Validation Loss: {best_val_loss} | Best Epoch: {results[best_fold-1]['f1'].best_epoch}")

    # Save metrics
    Logger.info("Saving metrics...")
    msg = om.save_object(results, cl.config.folder, dm.FolderType.results, "results.pkl" )
    Logger.info(msg)

    # Info
    Logger.info("Average Scores:")
    
    # TO DO: Add option to use warn metrics
    #use_warn_score = cl.config.metrics.use_warn_metrics

    Logger.info(f"Training Average-Scores: {get_mean_scores([v['f1'] for k, v in results.items()], 'train' )}")
    Logger.info(f"Validation Average-Scores: {get_mean_scores([v['f1'] for k, v in results.items()], 'val')}")

    # End LOSOCV
    end_train = datetime.datetime.now()
    Logger.info(f"Cross-Validation End time: {end_train}")
    Logger.info(f"Cross-Validation Duration: {end_train - start}")

    ###############################################
    ################ Inference ####################
    ###############################################
    Logger.info("Inference started...")
    print("======"*10)
    print("Inference started...")

    best_state = results[best_fold]['f1']
    best_state_l = results[best_fold]['loss']

    Logger.info("Creating Inference Dataset...")
    # Inference dataset
    X_inference = np.concatenate([X_dict[subject] for subject in inference_subjects], axis=0)
    y_in = np.concatenate([y_dict[subject] for subject in inference_subjects], axis=0)

    del X_dict, y_dict
    gc.collect()

    if cl.config.dataset.sensor == "acc":
        X_in = X_inference[:, :, :3]
        Logger.info("Using accelerometer data only.")
    elif cl.config.dataset.sensor == "gyro":
        X_in = X_inference[:, :, 3:]
        Logger.info("Using gyroscope data only.")
    else:
        X_in = X_inference
    
    Logger.info(f"Inference data shape: {X_in.shape} | Inference labels shape: {y_in.shape}")
    # Scale
    if best_state.scaler:
        X_in = best_state.scaler.transform(X_in.reshape(-1, num_features)).reshape(-1, window_size, num_features)

    inference_dataset = TensorDataset(torch.from_numpy(X_in).float(), torch.from_numpy(y_in).float())

    Logger.info(f"Inference dataset size: {len(inference_dataset)} | Sample shape: {inference_dataset[0][0].shape}")
    
   
    # Inference dataloader
    inference_loader = dp.load_dataloader(inference_dataset)

    # Load best criterion
    loss_fn = t.load_criterion(best_state.best_criterion_weight)

    # Inference
    inferece_metrics = t.run_epoch(0,"inference [F1-Score]", inference_loader,
                                    best_state.best_model,loss_fn,
                                    best_state.best_optimizer, best_state.best_lr_scheduler,
                                    device=device, is_binary= is_binary, threshold=cl.config.train.binary_threshold)[0]

    inferece_metrics.info()

    # Save inference metrics
    msg = om.save_object(inferece_metrics, cl.config.folder, dm.FolderType.results, "inference_metrics.pkl" )
    Logger.info(msg)
    
    # Inferece duration
    end_inference = datetime.datetime.now()
    Logger.info(f"Inference End time: {end_inference}")
    Logger.info(f"Inference Duration: {end_inference - end_train}")
    print(f"Inference End time: {end_inference}")
    print(f"Inference Duration: {end_inference - end_train}")

    ###############################################
    # Comparing best fold with best val loss
    
    if best_state_l.scaler:
        X_in = best_state_l.scaler.transform(X_inference.reshape(-1, num_features)).reshape(-1, window_size, num_features)
    
    inference_dataset = TensorDataset(torch.from_numpy(X_in).float(), torch.from_numpy(y_in).float())
    inference_loader =  dp.load_dataloader(inference_dataset)
    loss_fn = t.load_criterion(best_state_l.best_criterion_weight)
    inferece_metrics_l = t.run_epoch(0,"inference [Loss]", inference_loader,
                                    best_state_l.best_model,loss_fn,   
                                    best_state_l.best_optimizer, best_state_l.best_lr_scheduler,
                                    device=device, is_binary= is_binary, threshold=cl.config.train.binary_threshold)[0]

    Logger.info("Based on Val Loss:")
    inferece_metrics_l.info()

    del X_in, y_in, X_dict, y_dict, X_inference, inference_dataset, inference_loader, loss_fn
    gc.collect()
    
    # Results
    Logger.info("Results:")
    Logger.info("Folder: {cl.config.folder}")
    Logger.info(f"[Based on F1-Score] Best k-Fold: {best_fold+1}\nTrain F1-Score: {best_state.val_metrics.f1_score} | Val F1-Score: {best_state.val_metrics.f1_score} | Inference F1-Score: {inferece_metrics.f1_score}\nTrain Loss: {best_state.train_metrics.loss} | Val Loss: {best_state.val_metrics.loss} | Best Epoch: {best_state.best_epoch}")
    Logger.info(f"[Based on Loss] Best k-Fold: {best_fold_l+1}\nTrain F1-Score: {best_state_l.val_metrics.f1_score} | Val F1-Score: {best_state_l.val_metrics.f1_score} | Inference F1-Score: {inferece_metrics_l.f1_score}\nTrain Loss: {best_state_l.train_metrics.loss} | Val Loss: {best_state_l.val_metrics.loss} | Best Epoch: {best_state_l.best_epoch}")

    print("======"*10)
    print("Results:")
    print("Folder: {cl.config.folder}")
    print(f"[Based on F1-Score] Best k-Fold: {best_fold+1}\nTrain F1-Score: {best_state.val_metrics.f1_score} | Val F1-Score: {best_state.val_metrics.f1_score} | Inference F1-Score: {inferece_metrics.f1_score}\nTrain Loss: {best_state.train_metrics.loss} | Val Loss: {best_state.val_metrics.loss} | Best Epoch: {best_state.best_epoch}")
    print(f"[Based on Loss] Best k-Fold: {best_fold_l+1}\nTrain F1-Score: {best_state_l.val_metrics.f1_score} | Val F1-Score: {best_state_l.val_metrics.f1_score} | Inference F1-Score: {inferece_metrics_l.f1_score}\nTrain Loss: {best_state_l.train_metrics.loss} | Val Loss: {best_state_l.val_metrics.loss} | Best Epoch: {best_state_l.best_epoch}")

'''
def loso_cv(device):
    # start
    start = datetime.datetime.now()
    Logger.info(f"Cross-Validation Start time: {start}")

    # Empty dict to store results
    results = {}
    best_val_loss = np.inf
    best_subject = None

    # Get grouped files
    grouped_files = dp.get_files()

    # Split train and inference subjects
    train_subjects, inference_subjects = dp.split_data(grouped_files)
    Logger.info(f"Train subjects: {train_subjects}")
    Logger.info(f"Inference subjects: {inference_subjects}")

    # Get train and inference dataframes
    train_df = dfm.load_dfs_from(train_subjects, grouped_files, add_sub_id=True)
    inference_df = dfm.load_dfs_from(inference_subjects, grouped_files, add_sub_id=True)

    Logger.info(f"Train dataframe shape: {train_df.shape}")
    Logger.info(f"Inference dataframe shape: {inference_df.shape}")

    # Bandpass filter parameters
    order = cl.config.filter.order
    fc_high = cl.config.filter.fc_high
    fc_low = cl.config.filter.fc_low
    columns = train_df.filter(regex='acc*|gyro*').columns.tolist()
    fs = cl.config.filter.sampling_rate    
    # Apply Filtering
    train_df_filtered = band_pass_filter(train_df, order, fc_high, fc_low, columns, fs)
    inference_df_filtered = band_pass_filter(inference_df, order, fc_high, fc_low, columns, fs)

    Logger.info(f"Train dataframe shape after filtering: {train_df_filtered.shape} | Inference dataframe shape after filtering: {inference_df_filtered.shape}")
    #train_df_filtered = train_df
    #inference_df_filtered = inference_df


    # Loop through training subjects
    for subject in train_subjects:
        Logger.info(f"Validation on subject: {subject}")
        k_train_df = train_df_filtered[train_df_filtered['sub_id'] != subject].copy()
        k_val_df = train_df_filtered[train_df_filtered['sub_id'] == subject].copy()
        
        # Scale dataframes
        Logger.info("Scaling dataframes...")
        scaler = dp.get_scaler()

        k_train_scaled = scaler.fit_transform(k_train_df[columns])
        k_train_df[columns] = k_train_scaled
        k_train_df.reset_index(drop=True, inplace=True)

        k_val_scaled = scaler.transform(k_val_df[columns])
        k_val_df[columns] = k_val_scaled
        k_val_df.reset_index(drop=True, inplace=True)


        # Get train and val datasets
        Logger.info("Creating datasets...")
        train_dataset = dp.create_dataset(k_train_df)
        val_dataset = dp.create_dataset(k_val_df)  

        Logger.info(f"Train dataset size: {len(train_dataset)} | Val dataset size: {len(val_dataset)} | Sample shape: {train_dataset[0][0].shape}")

        # Del dataframes
        del k_train_df, k_val_df, k_train_scaled, k_val_scaled
        gc.collect()

        # Create data loaders
        Logger.info("Creating dataloaders...")
        train_loader = dp.load_dataloader(train_dataset)
        val_loader = dp.load_dataloader(val_dataset)
        

        # Compute weights
        Logger.info("Computing class weights...")   
        class_weights = dp.compute_weights(train_dataset)
        
        # Train model
        Logger.info("Loading model...")
        # Load Traning parameters
        model = t.load_network()
        optimizer = t.load_optim(model)
        criterion = t.load_criterion(class_weights)
        lr_scheduler = t.load_lr_scheduler(optimizer)

        Logger.info("Training model...")
        # Train Model
        state = t.train_model(model, criterion, 
                            optimizer, lr_scheduler,
                            train_loader, val_loader, device, optional_name="_cv-" + str(subject))

        if state.best_val_metrics.loss < best_val_loss:
            best_val_loss = state.best_val_metrics.loss
            best_subject = subject
            Logger.info(f"New best subject: {best_subject} with validation Loss: {best_val_loss} | F1-score:{state.best_val_metrics.f1_score}")


        state.info()

        # Visuals
        state.plot_losses(title=" Cross-Validation on subject: " + str(subject))
        state.plot_f1_scores(title=" Cross-Validation on subject: " + str(subject))

        # Save state to dict
        results[subject] = state

    Logger.info(f"Best Subject: {best_subject} with validation Loss: {best_val_loss}")
    Logger.info(f"Train subjects : {train_subjects}")

    # Save metrics
    Logger.info("Saving metrics...")
    msg = om.save_object(results, cl.config.folder, dm.FolderType.results, "results.pkl" )
    Logger.info(msg)

    # Info
    Logger.info("Average Scores:")
    Logger.info(f"Training Average-Scores: {get_mean_scores(results.values(), 'train')}")
    Logger.info(f"Validation Average-Scores: {get_mean_scores(results.values(), 'val')}")

    # End LOSOCV
    end_train = datetime.datetime.now()
    Logger.info(f"Cross-Validation End time: {end_train}")
    Logger.info(f"Cross-Validation Duration: {end_train - start}")

    ###############################################
    ################ Inference ####################
    ###############################################

    best_state = results[best_subject]

    Logger.info("Creating Inference Dataset...")
    # Inference dataset
    inference_dataset = dp.create_dataset(inference_df_filtered)

    Logger.info("Creating Inference Dataloader...")
    # Inference dataloader
    inference_loader = dp.load_dataloader(inference_dataset)

    Logger.info("Inference started...")
    # Load best criterion
    loss_fn = t.load_criterion(best_state.best_criterion_weight)

    # Inference
    inferece_metrics = t.run_epoch(0,"inference", inference_loader,
                                    best_state.best_model,loss_fn,
                                    best_state.best_optimizer, best_state.best_lr_scheduler,
                                    device=device)[0]

    inferece_metrics.info()

    # Save inference metrics
    msg = om.save_object(inferece_metrics, cl.config.folder, dm.FolderType.results, "inference_metrics.pkl" )
    Logger.info(msg)
    
    # Inferece duration
    end_inference = datetime.datetime.now()
    Logger.info(f"Inference End time: {end_inference}")
    Logger.info(f"Inference Duration: {end_inference - end_train}")

    '''
