import datetime
import gc
import numpy as np

from src.dl_pipeline import train as t
from src.helper import data_preprocessing as dp
from src.helper import df_manager as dfm
from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
from src.helper.filters import band_pass_filter
from src.helper import object_manager as om
from src.helper import directory_manager as dm
from src.helper.state import State

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

    if phase == "train":

        for state in states:
            f1_scores.append(state.best_train_metrics.f1_score)
            recall_scores.append(state.best_train_metrics.recall_score)
            precision_scores.append(state.best_train_metrics.precision_score)
            specificity_scores.append(state.best_train_metrics.specificity_score)
            jaccard_scores.append(state.best_train_metrics.jaccard_score)
            accuracy_scores.append(state.best_train_metrics.accuracy)
    else:
        for state in states:
            f1_scores.append(state.best_val_metrics.f1_score)
            recall_scores.append(state.best_val_metrics.recall_score)
            precision_scores.append(state.best_val_metrics.precision_score)
            specificity_scores.append(state.best_val_metrics.specificity_score)
            jaccard_scores.append(state.best_val_metrics.jaccard_score)
            accuracy_scores.append(state.best_val_metrics.accuracy)
    
    mean_scores = { "f1_score": np.mean(f1_scores),
                    "recall_score": np.mean(recall_scores), 
                    "precision_score": np.mean(precision_scores),
                    "specificity_score": np.mean(specificity_scores),
                    "jaccard_score": np.mean(jaccard_scores),
                    "accuracy": np.mean(accuracy_scores)
                    }

    return mean_scores
