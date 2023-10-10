from tensorflow import keras
from tensorflow.keras import layers

from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.helper import data_preprocessing as dp
import numpy as np
from collections import Counter
import gc


def load_data():
    is_binary = cl.config.dataset.num_classes < 3
    shelf_name = cl.config.dataset.name
    random_seed = cl.config.dataset.random_seed
    n_splits = cl.config.train.cross_validation.k_folds
    shuffle = cl.config.dataset.shuffle
    new_seed = random_seed if shuffle else None
    train_ratio = cl.config.dataset.train_ratio

    # Load python dataset
    X_dict, y_dict = dp.load_shelves(shelf_name)

    # All subjects
    subjects = list(X_dict.keys())
    
    X_all =  np.concatenate([X_dict[subject] for subject in subjects], axis=0)
    y_all =  np.concatenate([y_dict[subject] for subject in subjects], axis=0)

    # Reshape
    num, window_size, num_features = X_all.shape
    X_all = X_all.reshape(num, -1)

    # Split data
    if shuffle:
        X_train, X_inference, y_train, y_inference = train_test_split(X_all, y_all, train_size = train_ratio, stratify = y_all, shuffle=shuffle, random_state = new_seed)
    else:
        X_train, X_inference, y_train, y_inference = train_test_split(X_all, y_all, train_size = train_ratio)

    X_train = X_train.reshape(-1, window_size, num_features)
    X_inference = X_inference.reshape(-1, window_size, num_features)

    Logger.info(f"Total Train size: {len(X_train)} | Counts: {Counter(y_train)}")
    Logger.info(f"Inference size: {len(X_inference)} | Counts: {Counter(y_inference)}")

    del X_dict, y_dict
    gc.collect()

    return X_train, X_inference, y_train, y_inference

# Encoder
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# Model
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes=3):

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    out_dim = 1 if n_classes == 2 else n_classes
    outputs = layers.Dense(out_dim, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_optim(optim, lr, weight_decay):
    if optim == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr,
        weight_decay=weight_decay)
    elif optim == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=lr,
        weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr,
        weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=lr,
        weight_decay=weight_decay)
    elif optim == 'adagrad':
        optimizer = keras.optimizers.Adagrad(learning_rate=lr,
        weight_decay=weight_decay)
    elif optim == 'adadelta':
        optimizer = keras.optimizers.Adadelta(learning_rate=lr,
        weight_decay=weight_decay)
    elif optim == 'adamax':
        optimizer = keras.optimizers.Adamax(learning_rate=lr,
        weight_decay=weight_decay)
    elif optim == 'nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=lr,
        weight_decay=weight_decay)
    elif optim == 'ftrl':
        optimizer = keras.optimizers.Ftrl(learning_rate=lr,
        weight_decay=weight_decay)
    elif otpim == 'adafactor':
        optimizer = keras.optimizers.Adafactor(learning_rate=lr,
        weight_decay=weight_decay)
    else:
        optimizer = keras.optimizers.AdamW(learning_rate=lr,
        weight_decay=weight_decay)

    return optimizer

def main(config):
    
    num_classes = cl.config.dataset.num_classes

    # Load data
    x_train, x_test, y_train, y_test = load_data()
    input_shape = x_train.shape[1:]

    # Transformer Layers
    hidden_channels = config.get("hidden_channels", [32, 64, 128])
    kernel_sizes = config.get("kernel_sizes", [5, 5, 5])

    nhead = config.get("multi_attn_heads", 3)
    dim_feedforward = config.get("dim_feedforward", 32)
    transformer_dropout = config.get("transformer_dropout", 0.25)
    transformer_act_fn = config.get("transformer_act_fn", 'relu')
    num_encoders = config.get('num_encoder_layers', 2)
    num_decoders = config.get('num_decoder_layers', 2)
    encode_position = config.get('encode_position', True)
    transformer_dim = hidden_channels[-1]
    fc_hidden_size = config.get('fc_hidden_size', 128)
    fc_dropout = config.get('dropout', 0.25)

    model = build_model(
            input_shape,
            head_size=transformer_dim,
            num_heads=nhead,
            ff_dim=dim_feedforward,
            num_transformer_blocks=num_encoders,
            mlp_units=[fc_hidden_size],
            mlp_dropout=fc_dropout,
            dropout=transformer_dropout,
            n_classes=num_classes
            )

   

    # Class weights
    class_weights = dp.compute_weights(y_train)

    # Loss function
    if num_classes == 2:
        loss = keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Optimizer
    lr = cl.config.optim.learning_rate
    weight_decay = cl.config.optim.weight_decay
    optim = cl.config.optim.name
    
    optimizer = get_optim(optim, lr, weight_decay)
    #f1_metric = keras.metrics.F1Score(average="macro")
    auc_metric = keras.metrics.AUC(multi_label=num_classes > 2)
    sparse_acc = keras.metrics.SparseCategoricalAccuracy()

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics= [auc_metric, sparse_acc]
    )

    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    # Train the model
    batch_size = cl.config.train.batch_size
    epochs = cl.config.train.num_epochs
    val_split = cl.config.dataset.validation_split
    shuffle = cl.config.dataset.shuffle
    num_workers = cl.config.dataset.num_workers

    model.fit(
        x_train,
        y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        #verbose=1,
        #use_multiprocessing=True,
        #workers=num_workers,
        #class_weight = class_weights,
        #shuffle=shuffle
    )

    # Evaluate the trained model
    model.evaluate(x_test, y_test, verbose=1)
