# ------------------------------------------------------------------------
# Description: Train Module
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import os
import sys
import datetime
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.helper.metrics import Metrics
from src.helper.logger import Logger
from src.helper.state import State
from src.utils.config_loader import config_loader as cl
from src.dl_pipeline.architectures.CNN import CNNModel
from src.dl_pipeline.architectures.LSTMs import DeepConvLSTM

# Function to save state of the model
def save_state(state:State, optional_name:str = ""):
    #todo: check
    filename = cl.config.architecture.name + optional_name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pth'
    
    model_path = cl.config.models_path + "/" + filename

    if not os.path.exists(cl.config.models_path):
        os.makedirs(cl.config.models_path)
    
    state_dict = {
        'path':model_path,
        'filename':filename,
        'epoch': state.best_epoch, 
        'state_dict': state.best_model.state_dict(),
        'optimizer': state.best_optimizer.state_dict(),
        'train_metrics': state.best_train_metrics, 
        'val_metrics': state.best_val_metrics,
        'criterion': state.best_criterion_weight
        }

    if state.best_lr_scheduler:
        state_dict['lr_scheduler'] = state.best_lr_scheduler.state_dict()
            
    #Save model    
    try:
        torch.save(state, model_path)
    except Exception as e:  
        print(f"An error occurred: {str(e)}")
        return
    
    print(f"Model saved at {model_path}")
    
# Function to load state of the model
# todo: check
def load_checkpoint(model, filename, optimizer= None, lr_scheduler=None):
    full_path = os.path.join(cl.config.models_path,filename)
    if os.path.isfile(full_path):
        print("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(full_path)
        
        # Set state object
        state = State()
        state.set_file_name(filename)
        state.set_path(full_path)
        state.best_epoch = checkpoint['epoch']
        state.best_model = model.load_state_dict(checkpoint['state_dict'])
        state.best_train_metrics = checkpoint['train_metrics']
        state.best_val_metrics = checkpoint['val_metrics']
        state.best_criterion_weight = checkpoint['criterion']

        if optimizer != None:
            state.best_optimizer= optimizer.load_state_dict(checkpoint['optimizer'])
        
        if lr_scheduler != None:
            try:
                state.best_lr_scheduler = lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("No lr_scheduler found in checkpoint")
                state.best_lr_scheduler = None

        print("Loaded checkpoint '{}' (Epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("No checkpoint found at '{}'".format(filename))
        
    return state


# Function to run one epoch of training and validataion
def run_epoch(epoch, phase, data_loader, network, criterion, optimizer, lr_scheduler, device, is_binary, threshold=0.5):
    """Runs one epoch of training and validation and returns network, criterion, optimizer and lr_scheduler.

    Args:
        epoch (int): Epoch number
        phase (str): _description_
        data_loader (dataloader): _description_
        network (_type_): _description_
        criterion (_type_): _description_
        optimizer (_type_): _description_
        lr_scheduler (_type_): _description_
        args (_type_): _description_
        Logger (_type_): _description_
        epoch_start_time (_type_): _description_

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """

    is_train = phase == 'train'
    
    # Set network to train or eval mode
    if is_train:
        network.train()
    else:
        network.eval()

    # Initialize lists for running_loss, targets, predictions
    running_loss = 0.0
    epoch_targets = np.array([])
    epoch_preds = np.array([])

    # Iterate over data_loader
    for batch_idx, (X, y) in enumerate(data_loader):
        # Mean zero center x
        #x = x - torch.mean(x, dim=(2, 3), keepdim=True)
        #X, y = X.double(), y.double()
        # Send x and y to GPU/CPU
        if not is_binary:
            y = y.long()

        inputs, targets = X.to(device), y.to(device)

        # Zero accumulated gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            # Pass data to forward pass of the network.
            output = network(inputs)  
            
            if is_binary:
                output = output.squeeze(1)

             #Find loss
            loss = criterion(output, targets)            
            
            if is_train:
                loss.backward() # backward pass of network (calculate sum of gradients for graph)
                optimizer.step() # perform model parameter update (update weights)
            
            # TODO: Try MaxUp   loss

            #Find predicted label
            # TODO: logits vs activation on Neural Network
            
            if is_binary:
                prediction = (torch.sigmoid(output) > threshold).float()
            else:
                prediction = torch.argmax(output, dim=1) # dim=1 because dim=0 is batch size
            
        #Sum loss
        running_loss += loss.item() * targets.size(0)
        
        #Append labels
        epoch_targets = np.concatenate((np.array(epoch_targets, 'uint8'), np.array(targets.detach().cpu(), 'uint8')))
        
        #Append Predictions
        epoch_preds = np.concatenate((np.array(epoch_preds, 'uint8'), np.array(prediction.detach().cpu(), 'uint8')))

    #Length of dataset
    dataset_len = len(epoch_preds)
    
    #Calculate epoch loss
    epoch_loss = running_loss / dataset_len # type: ignore
    
    # Lr scheduler step in train phase
    if lr_scheduler and is_train:
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(np.mean(epoch_loss))
            last_lr = optimizer.param_groups[0]['lr']
        else:
            lr_scheduler.step()
            last_lr = lr_scheduler.get_last_lr()   
        Logger.info(f"LR Update in Epoch: {epoch+1}/{cl.config.train.num_epochs} | last_lr: {last_lr}")

    #Calculate metrics
    metrics = Metrics(epoch, is_binary)
    metrics.phase = phase
    metrics.y_true = epoch_targets
    metrics.y_pred = epoch_preds
    metrics.calculate_metrics()
    metrics.set_loss(epoch_loss)

    #Empty Cuda Cache
    if device == 'cuda':
        torch.cuda.empty_cache()

    #Return epoch loss and metrics
    return metrics, network, criterion, optimizer, lr_scheduler

   
def train_model(network, criterion, optimizer, lr_scheduler, train_loader, val_loader,device, is_binary, optional_name:str = "", threshold=0.5):
    """Trains the model and returns the trained model, criterion, optimizer and lr_scheduler.

    Args:
        network (_type_): _description_
        criterion (_type_): _description_
        optimizer (_type_): _description_
        lr_scheduler (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
        args (_type_): _description_
        Logger (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Start time    
    start = datetime.datetime.now() 

    # Init state
    state = State()

    # Best F1 score
    #best_f1_score = 0.0

    # Best loss
    best_val_loss = np.inf

    # Get epochs from config file
    try:
        epochs = int(cl.config.train.num_epochs)
    except AttributeError as e:
        Logger.warning(f"Total number of epochs is not set in config file. Using 100.")
        epochs = 100
    
    # Initialize lists for train and val metrics
    train_metrics_arr = []
    val_metrics_arr = []

    # Start training loop
    for epoch in tqdm(range(epochs),desc="Training model:"):
        
        # Run training phase
        train_metrics, network, criterion, optimizer, lr_scheduler = run_epoch(epoch, 'train', train_loader, network, criterion, optimizer, lr_scheduler,device, is_binary, threshold)
        
        # Append train_metrics
        train_metrics_arr.append(train_metrics)

        # Run validation phase
        val_metrics, network, criterion, optimizer, lr_scheduler = run_epoch(epoch, 'val', val_loader, network, criterion, optimizer, lr_scheduler,device, is_binary, threshold)
        
        # Append val_metrics
        val_metrics_arr.append(val_metrics)

        # Print epoch results
        train_message = f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_metrics.loss:.2f} | Train F1 Score: {train_metrics.f1_score:.2f} | Train Recall Score: {train_metrics.recall_score:.2f} | Train Precision Score: {train_metrics.precision_score:.2f} | Train Specificity Score: {train_metrics.specificity_score:.4f} | Train Jaccard Score: {train_metrics.jaccard_score:.2f}"
        val_message = f"Epoch: {epoch+1}/{epochs} | Val Loss: {val_metrics.loss:.2f} | Val F1 Score: {val_metrics.f1_score:.2f} | Val Recall Score: {val_metrics.recall_score:.2f} | Val Precision Score: {val_metrics.precision_score:.2f} | Val Specificity Score: {val_metrics.specificity_score:.4f} | Val Jaccard Score: {val_metrics.jaccard_score:.2f}"

        #message = f"Epoch:{epoch+1}/{epochs} | Train Loss: {train_metrics.loss:.2f}"
        Logger.info(train_message)
        Logger.info(val_message)

        # save best model
        if val_metrics.loss < best_val_loss: 
            Logger.info(f"Saving best model with Train Loss: {train_metrics.loss:.2f} | Val Loss: {val_metrics.loss} | F1 score: {val_metrics.f1_score:.2f} | Epoch: {epoch+1}/{epochs}")
            
            best_val_loss = val_metrics.loss
    
            state.best_epoch = epoch + 1
            state.best_model = network
            state.best_optimizer = optimizer
            state.best_criterion_weight = criterion.weight
            state.best_train_metrics = train_metrics
            state.best_val_metrics = val_metrics
            
            if lr_scheduler is not None:
                state.best_lr_scheduler = lr_scheduler
            
            # Save state
            save_state(state, optional_name)

    state.train_metrics_arr = train_metrics_arr
    state.val_metrics_arr = val_metrics_arr
    
    # End time
    end = datetime.datetime.now()

    # Print total time taken
    Logger.info(f"Total time taken: {end-start}")

    return state


# Function to load network
def load_network():
    # TODO: Check for other networks
    network = cl.config.architecture.name

    if network == "cnn":
        # TO DO:
        raise NotImplementedError
        model = CNNModel(cl.config_dict['architecture'])
    
    elif network == "deepconvlstm":
        model = DeepConvLSTM(cl.config_dict['architecture'])
    else:
        return None
    
    Logger.info(f"Using Model: {model}")
    # Return initialized model
    return model.apply(init_weights)

# Function to load optimizer
def load_optim(model):
    optim_name = cl.config.optim.name
    lr = cl.config.optim.learning_rate
    momentum = cl.config.optim.momentum
    weight_decay = cl.config.optim.weight_decay

    if optim_name == "adam":
        Logger.info(f"Using Adam optimizer with params: lr={lr}, weight_decay={weight_decay}")
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "sgd":
        Logger.info(f"Using SGD optimizer with params: lr={lr}, momentum={momentum}, weight_decay={weight_decay}")
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,momentum=momentum)

# Function to load criterion
def load_criterion(weights):
    loss = cl.config.criterion.name

    if cl.config.criterion.weighted and weights is not None:
        class_weights = weights.to(cl.config.train.device)
        if loss == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            Logger.info(f"Using CrossEntropyLoss with class weights: {class_weights}")
        elif loss == 'bce':
            # Pass positive class weight only
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
            Logger.info(f"Using BCEWithLogitsLoss with class weights: {class_weights}")
    else:
        if cl.config.dataset.num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
            Logger.info(f"Using BCEWithLogitsLoss")
        else:    
            criterion = nn.CrossEntropyLoss()
            Logger.info(f"Using CrossEntropyLoss")
    return criterion

# Function to load lr_scheduler
def load_lr_scheduler(optimizer):
    scheduler = cl.config.lr_scheduler.name
    verbose = cl.config.lr_scheduler.verbose

    if scheduler == "step_lr":
        step_size = cl.config.lr_scheduler.step_size
        gamma = cl.config.lr_scheduler.gamma
        Logger.info(f"Using StepLR with params: step_size={step_size}, gamma={gamma}")
        return StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=verbose)
    elif scheduler == "reduce_lr_on_plateau":
        mode = cl.config.lr_scheduler.mode
        factor = cl.config.lr_scheduler.factor
        patience = cl.config.lr_scheduler.patience
        threshold = cl.config.lr_scheduler.threshold
        Logger.info(f"Using ReduceLROnPlateau with params: mode:{mode} | factor:{factor} | patience:{patience}| threshold: {threshold}")
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)
    else:
        Logger.info("No lr_scheduler found. Returning None.")
        return None


# Function to init weights
def init_weights(model):
    """
    Initialize weights of the DL network.

    Args:
        model (nn.Module): DL network
    
    Returns:
        DL network with initialized weights.
    """

    scheme = cl.config.architecture.scheme

    # Loop though all modules
    for module in model.modules():

        # Linear layers
        if isinstance(module, nn.Linear):

            if scheme == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif scheme == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif scheme == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight)
            elif scheme == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight)
            elif scheme == "orthogonal":
                nn.init.orthogonal_(module.weight)
            elif scheme == "normal":
                nn.init.normal_(module.weight)
            else:
                Logger.error(f"Invalid weight initialization scheme: {scheme}")
                raise ValueError

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            
        # Convolutional layers
        elif isinstance(module, nn.Conv2d):

            if scheme == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif scheme == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif scheme == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight)
            elif scheme == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight)
            elif scheme == "orthogonal":
                nn.init.orthogonal_(module.weight)
            elif scheme == "normal":
                nn.init.normal_(module.weight)
            else:
                Logger.error(f"Invalid weight initialization scheme: {scheme}")
                raise ValueError

            if module.bias is not None:
                module.bias.data.fill_(0.00)
        
        # BatchNorm layers
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.fill_(0.0)
        
        # LSTM layers
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name or 'weight_hh' in name:
                    if scheme == "xavier_uniform":
                        nn.init.xavier_uniform_(param)
                    elif scheme == "xavier_normal":
                        nn.init.xavier_normal_(param)
                    elif scheme == "kaiming_uniform":
                        nn.init.kaiming_uniform_(param)
                    elif scheme == "kaiming_normal":
                        nn.init.kaiming_normal_(param)
                    elif scheme == "orthogonal":
                        nn.init.orthogonal_(param)
                    elif scheme == "normal":
                        nn.init.normal_(param)
                    else:
                        Logger.error(f"Invalid weight initialization scheme: {scheme}")
                        raise ValueError
        
        return model